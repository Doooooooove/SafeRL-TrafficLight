from pathlib import Path
from math import dist
import shlex

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sumo_rl
from sumo_rl.environment.env import LIBSUMO
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
from traci import constants as tc


class _SumoArguments(list):
    """SUMO-RL 1.4.5 calls split() when passing additional options to TraCI."""

    def split(self):
        return self


class _CountingEnvironment(sumo_rl.SumoEnvironment):
    def __init__(self, *args, program_id=None, **kwargs):
        self.program_id = program_id
        self.interval_arrivals = 0
        self.vehicle_sample = None
        super().__init__(*args, **kwargs)

    def _start_simulation(self):
        super()._start_simulation()
        if self.program_id is not None:
            self.sumo.trafficlight.setProgram(self.ts_ids[0], self.program_id)
        if not LIBSUMO:
            # One subscription covers this intersection and every lane in its network.
            lower, upper = self.sumo.simulation.getNetBoundary()
            self.sumo.junction.subscribeContext(
                self.ts_ids[0], tc.CMD_GET_VEHICLE_VARIABLE, dist(lower, upper) + 1.0,
                (tc.VAR_SPEED, tc.VAR_WAITING_TIME, tc.VAR_ACCELERATION),
            )

    def _compute_observations(self):
        if not LIBSUMO:
            self.vehicle_sample = self.sumo.junction.getContextSubscriptionResults(self.ts_ids[0])
        return super()._compute_observations()

    def _get_system_info(self):
        if self.vehicle_sample is None:
            return super()._get_system_info()
        vehicles = self.sumo.vehicle.getIDList()
        if any(v not in self.vehicle_sample for v in vehicles):
            return super()._get_system_info()
        speeds = [self.vehicle_sample[v][tc.VAR_SPEED] for v in vehicles]
        waits = [self.vehicle_sample[v][tc.VAR_WAITING_TIME] for v in vehicles]
        return {
            "system_total_stopped": sum(speed < 0.1 for speed in speeds),
            "system_total_waiting_time": sum(waits),
            "system_mean_waiting_time": np.mean(waits) if waits else 0.0,
            "system_mean_speed": np.mean(speeds) if speeds else 0.0,
        }

    def _sumo_step(self):
        super()._sumo_step()
        # SUMO reports arrivals in the last simulation step, not a running total.
        self.interval_arrivals += self.sumo.simulation.getArrivedNumber()


class UnifiedIntersectionObservation(ObservationFunction):
    """Keep the pretrained layout: queues, headways, green phase, timer, approach risks."""

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)
        self.max_headway = 50.0
        self.ttc_threshold = 2.0
        self.conflict_count = 0
        self.hard_brakes = 0

    def __call__(self) -> np.ndarray:
        queues = self.ts.get_lanes_queue()
        headways = []
        risks = [0.0] * 4
        approaches = {"n_t": 0, "e_t": 1, "s_t": 2, "w_t": 3}
        self.conflict_count = 0
        self.hard_brakes = 0
        vehicle = self.ts.sumo.vehicle
        samples = getattr(self.ts.env, "vehicle_sample", None) or {}
        speeds = {v: row[tc.VAR_SPEED] for v, row in samples.items()}

        for lane in self.ts.lanes:
            vehicles = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            gaps = []
            for i, veh_id in enumerate(vehicles):
                acc = samples[veh_id][tc.VAR_ACCELERATION] if veh_id in samples else vehicle.getAcceleration(veh_id)
                self.hard_brakes += int(acc < -4.5)
                leader = vehicle.getLeader(veh_id)
                if not leader or not leader[0]:
                    continue
                leader_id, distance = leader
                if i + 1 < len(vehicles) and leader_id == vehicles[i + 1]:
                    gaps.append(min(distance, self.max_headway))
                if distance <= 0:
                    continue
                for item in (veh_id, leader_id):
                    if item not in speeds:
                        speeds[item] = vehicle.getSpeed(item)
                closing_speed = speeds[veh_id] - speeds[leader_id]
                if closing_speed <= 0.1:
                    continue
                ttc = distance / closing_speed
                if ttc < self.ttc_threshold:
                    risks[approaches[lane[:3]]] += min(1.0, 1.0 / (ttc + 0.1))
                if ttc < 1.5 and closing_speed > 0.5:
                    self.conflict_count += 1
            headways.append(float(np.mean(gaps)) / self.max_headway if gaps else 1.0)

        if self.ts.env.fixed_ts:
            # Native programs alternate green and yellow; report the target
            # green during yellow, as SUMO-RL does in agent-controlled mode.
            signal = self.ts.sumo.trafficlight
            phase = ((signal.getPhase(self.ts.id) + 1) // 2) % self.ts.num_green_phases
            remaining = signal.getNextSwitch(self.ts.id) - self.ts.env.sim_step
        else:
            phase = self.ts.green_phase
            remaining = self.ts.max_green - self.ts.time_since_last_phase_change
        one_hot = [float(i == phase) for i in range(self.ts.num_green_phases)]
        ratio = float(np.clip(remaining / self.ts.max_green, 0.0, 1.0))
        risks = [min(1.0, value / 3.0) for value in risks]
        return np.asarray(queues + headways + one_hot + [ratio] + risks, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        dim = 2 * len(self.ts.lanes) + self.ts.num_green_phases + 1 + 4
        return spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)


class UnifiedMDPIntersectionEnv(gym.Env):
    """Single-intersection Gymnasium environment with interval throughput and sampled safety metrics."""

    def __init__(
        self,
        use_gui=False,
        num_seconds=20000,
        delta_time=5,
        min_green=5,
        max_green=50,
        w_efficiency=1.0,
        w_delay=0.1,
        w_conflict=5.0,
        enable_safety_shield=True,
        sumo_seed=42,
        additional_sumo_cmd=None,
        fixed_ts=False,
        program_id=None,
        begin_time=0,
        route_file=None,
    ):
        super().__init__()
        scenario = Path(__file__).resolve().parent / "scenario"
        if program_id is not None and not fixed_ts:
            raise ValueError("program_id requires fixed_ts=True for native signal control")
        if isinstance(additional_sumo_cmd, str):
            lexer = shlex.shlex(additional_sumo_cmd, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = lexer.escape = ""
            additional_sumo_cmd = list(lexer)
        if additional_sumo_cmd is not None:
            additional_sumo_cmd = _SumoArguments(map(str, additional_sumo_cmd))

        self.w_efficiency = w_efficiency
        self.w_delay = w_delay
        self.w_conflict = w_conflict
        self.enable_safety_shield = enable_safety_shield
        self.min_green = min_green
        self.env = _CountingEnvironment(
            net_file=str(scenario / "single-intersection.net.xml"),
            route_file=str(route_file or scenario / "single-intersection-vhvh.rou.xml"),
            use_gui=use_gui,
            begin_time=begin_time,
            num_seconds=num_seconds,
            single_agent=True,
            delta_time=delta_time,
            min_green=min_green,
            max_green=max_green,
            observation_class=UnifiedIntersectionObservation,
            # Per-agent info already computes the same accumulated lane waits.
            reward_fn=lambda ts: 0.0,
            sumo_seed=sumo_seed,
            additional_sumo_cmd=additional_sumo_cmd,
            fixed_ts=fixed_ts,
            program_id=program_id,
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.ts_id = self.env.ts_ids[0]

    def get_action_mask(self) -> np.ndarray:
        ts = self.env.traffic_signals[self.ts_id]
        mask = np.ones(self.action_space.n, dtype=bool)
        if not self.env.fixed_ts and ts.time_since_last_phase_change < ts.yellow_time + self.min_green:
            mask[:] = False
            mask[ts.green_phase] = True
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_wait = info["agents_total_accumulated_waiting_time"] / 100.0
        return obs, info

    def step(self, action):
        ts = self.env.traffic_signals[self.ts_id]
        if self.enable_safety_shield and not self.env.fixed_ts:
            if not self.get_action_mask()[action]:
                action = ts.green_phase

        self.env.interval_arrivals = 0
        obs, _, terminated, truncated, info = self.env.step(action)
        step_throughput = self.env.interval_arrivals
        cur_wait = info["agents_total_accumulated_waiting_time"] / 100.0
        diff_waiting = self.last_wait - cur_wait
        self.last_wait = cur_wait
        sample = ts.observation_fn
        reward = (
            self.w_efficiency * step_throughput
            + self.w_delay * diff_waiting
            - self.w_conflict * sample.hard_brakes
        )
        info["step_throughput"] = step_throughput
        info["system_waiting_time"] = info["system_total_waiting_time"]
        info["conflict_count"] = sample.conflict_count
        info["hard_brakes"] = sample.hard_brakes
        info["action_mask"] = self.get_action_mask()
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()
