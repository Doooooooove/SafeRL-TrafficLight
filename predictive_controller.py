"""Choose the next green by forecasting queues over four decision intervals."""

from math import ceil

import numpy as np


class PredictiveController:
    horizon = 4
    discharge_rate = 0.48  # vehicles per second per green lane

    def __init__(self):
        self.signal = None

    def _prepare(self, signal):
        self.signal = signal
        lane_api = signal.sumo.lane
        lanes = signal.lanes
        self.speed_limits = [lane_api.getMaxSpeed(lane) for lane in lanes]
        self.previous_ids = [set() for _ in lanes]
        self.inflow = np.zeros(len(lanes))

        links = signal.sumo.trafficlight.getControlledLinks(signal.id)
        green_lanes = [
            {link[0] for state, group in zip(phase.state, links)
             if state in "gG" for link in group}
            for phase in signal.green_phases
        ]
        served = np.array([[lane in group for lane in lanes] for group in green_lanes])
        minimum_slots = ceil((signal.yellow_time + signal.min_green) / signal.delta_time)
        self.plans = {}
        for current in range(len(green_lanes)):
            # Extend each feasible signal sequence by one slot. Put holding
            # first so exactly tied forecasts avoid unnecessary changes.
            paths = [([], current, 0)]
            for _ in range(self.horizon):
                next_paths = []
                for path, last, locked in paths:
                    next_paths.append((path + [last], last, max(0, locked - 1)))
                    if locked == 0:
                        next_paths.extend(
                            (path + [phase], phase, minimum_slots - 1)
                            for phase in range(len(green_lanes)) if phase != last
                        )
                paths = next_paths
            phases = np.array([path for path, _, _ in paths])
            changed = np.diff(phases, axis=1, prepend=current) != 0
            green_seconds = signal.delta_time - changed * signal.yellow_time
            capacity = served[phases] * green_seconds[:, :, None] * self.discharge_rate
            self.plans[current] = phases[:, 0], capacity

    def predict(self, env):
        signal = env.env.traffic_signals[env.ts_id]
        if signal is not self.signal:
            self._prepare(signal)

        lanes = signal.lanes
        slot = signal.delta_time
        can_switch = env.get_action_mask().all()
        lane_api, vehicle_api = signal.sumo.lane, signal.sumo.vehicle
        queues = np.zeros(len(lanes))
        arrivals = np.zeros((self.horizon, len(lanes)))
        weights = np.ones(len(lanes))
        for i, lane in enumerate(lanes):
            ids = lane_api.getLastStepVehicleIDs(lane)
            current_ids = set(ids)
            entries = len(current_ids - self.previous_ids[i])
            self.inflow[i] = 0.85 * self.inflow[i] + 0.15 * entries
            self.previous_ids[i] = current_ids
            if not can_switch:
                continue

            queues[i] = lane_api.getLastStepHaltingNumber(lane)
            weights[i] += lane_api.getWaitingTime(lane) / max(1, queues[i]) / 40.0
            length = signal.lanes_length[lane]
            limit = self.speed_limits[i]
            for vehicle in ids:
                speed = vehicle_api.getSpeed(vehicle)
                if speed < 0.1:
                    continue  # Already counted in the initial queue.
                # Approximate the queue tail using six metres per stopped vehicle.
                distance = max(0, length - vehicle_api.getLanePosition(vehicle) - 6 * queues[i])
                eta = distance / max(speed, 0.5 * limit)
                interval = int(eta / slot)
                if interval < self.horizon:
                    arrivals[interval, i] += 1
            # Vehicles yet to enter can first reach the stop line after transit.
            transit_slots = round(length / limit / slot)
            arrivals[transit_slots:, i] += self.inflow[i]

        current = signal.green_phase
        if not can_switch:
            return current
        first_actions, capacity = self.plans[current]
        costs = np.zeros(len(first_actions))
        for interval in range(self.horizon):
            queues = np.maximum(0, queues + arrivals[interval] - capacity[:, interval])
            costs += (queues * weights).sum(axis=1)
        return int(first_actions[np.argmin(costs)])
