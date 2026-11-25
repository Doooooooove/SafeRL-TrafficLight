"""Score each green phase from current lane observations using a shared linear policy."""

import json
from pathlib import Path

import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "models" / "phase_policy.json"
LEGACY_FEATURES = ("stopped", "moving_near", "moving_far", "waiting", "paired_demand",
                   "continue_green", "time_since_service", "recent_entries")
FEATURES = (*LEGACY_FEATURES, "approaching", "continue_discharge", "queue_pressure",
            "continue_parallel_discharge")


class LearnedPhaseController:
    """Score legal phases using lane demand and the current episode's history."""

    def __init__(self, weights=None):
        if weights is None:
            saved = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
            names = LEGACY_FEATURES if set(saved) == set(LEGACY_FEATURES) else FEATURES
            weights = [saved[name] for name in names]
        self.weights = np.asarray(weights, dtype=float)
        self.legacy = len(self.weights) == len(LEGACY_FEATURES)
        self.signal = None

    def _initialize(self, signal):
        self.signal = signal
        links = signal.sumo.trafficlight.getControlledLinks(signal.id)
        self.phase_lanes = [
            tuple(sorted({link[0] for color, group in zip(phase.state, links)
                          if color in "Gg" for link in group}))
            for phase in signal.green_phases
        ]
        self.last_green = np.zeros(len(self.phase_lanes))
        self.entries = dict.fromkeys(signal.lanes, 0.0)
        self.previous = {lane: set() for lane in signal.lanes}
        if not self.legacy:
            self.lane_speeds = {lane: signal.sumo.lane.getMaxSpeed(lane) for lane in signal.lanes}
        self.elapsed = 0

    def predict(self, env):
        signal = env.env.traffic_signals[env.ts_id]
        return self.choose_phase(signal, env.get_action_mask().all())

    def choose_phase(self, signal, can_switch):
        """Update lane history on every decision, including during minimum green."""
        if signal is not self.signal:
            self._initialize(signal)
        current = signal.green_phase
        self.elapsed += signal.delta_time
        self.last_green[current] = self.elapsed
        stats = {}
        for lane in signal.lanes:
            vehicles = signal.sumo.lane.getLastStepVehicleIDs(lane)
            ids = set(vehicles)
            new_entries = len(ids - self.previous[lane])
            self.entries[lane] = 0.85 * self.entries[lane] + 0.15 * new_entries
            self.previous[lane] = ids
            if not can_switch:
                continue
            stopped = signal.sumo.lane.getLastStepHaltingNumber(lane)
            distances = [signal.lanes_length[lane] - signal.sumo.vehicle.getLanePosition(v)
                         for v in vehicles]
            near = sum(distance < 75 for distance in distances)
            # Allocate stopped vehicles to the near zone first. This is a
            # lane-level estimate and also handles queues longer than 75 m.
            moving_near = max(0, near - stopped)
            moving_far = len(ids) - stopped - moving_near
            wait = signal.sumo.lane.getWaitingTime(lane) / 30.0
            row = (stopped, moving_near, moving_far, wait, self.entries[lane])
            if not self.legacy:
                # Estimate time to the queue tail. Give most credit to vehicles that
                # can join the queue or cross during the next two decisions.
                distances.sort()
                tail = distances[stopped - 1] if stopped else 0.0
                slot_distance = signal.delta_time * self.lane_speeds[lane]
                approach = discharge = 0.0
                for distance in distances[stopped:]:
                    slots = max(0.0, distance - tail) / slot_distance
                    approach += max(0.0, 1.0 - slots / 2.0)
                    discharge += max(0.0, 1.0 - slots)
                # Cap at roughly one lane's discharge in a five-second green.
                row += (approach, min(2.5, stopped + discharge), max(0, stopped - 4))
            stats[lane] = row
        if not can_switch:
            return current

        scores = []
        for phase, lanes in enumerate(self.phase_lanes):
            lane_rows = [stats[lane] for lane in lanes]
            stopped, near, far, wait, entries, *forecast = map(sum, zip(*lane_rows))
            if stopped + near + far == 0:
                scores.append(0.0)
                continue
            # Opposite approaches can discharge in parallel on this network.
            paired = min(row[0] + row[1] for row in lane_rows)
            age = min(4.0, (self.elapsed - self.last_green[phase]) / 30.0)
            features = [stopped, near, far, wait, paired, int(phase == current), age, entries]
            if forecast:
                approach, discharge, pressure = forecast
                parallel = min(row[6] for row in lane_rows)
                features.extend((approach, discharge if phase == current else 0.0, pressure,
                                 parallel if phase == current else 0.0))
            scores.append(self.weights @ features)
        return max(range(len(scores)), key=lambda phase: (scores[phase], phase == current))
