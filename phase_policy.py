"""Score each green phase from current lane observations using a shared linear policy."""

import json
from pathlib import Path

import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "models" / "phase_policy.json"
FEATURES = ("stopped", "moving_near", "moving_far", "waiting", "paired_demand",
            "continue_green", "time_since_service", "recent_entries")


class LearnedPhaseController:
    """Score legal phases using lane demand and the current episode's history."""

    def __init__(self, weights=None):
        if weights is None:
            saved = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
            weights = [saved[name] for name in FEATURES]
        self.weights = np.asarray(weights, dtype=float)
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
            near = sum(signal.lanes_length[lane] - signal.sumo.vehicle.getLanePosition(v) < 75
                       for v in vehicles)
            # Allocate stopped vehicles to the near zone first. This is a
            # lane-level estimate and also handles queues longer than 75 m.
            moving_near = max(0, near - stopped)
            moving_far = len(ids) - stopped - moving_near
            wait = signal.sumo.lane.getWaitingTime(lane) / 30.0
            stats[lane] = stopped, moving_near, moving_far, wait, self.entries[lane]
        if not can_switch:
            return current

        scores = []
        for phase, lanes in enumerate(self.phase_lanes):
            lane_rows = [stats[lane] for lane in lanes]
            stopped, near, far, wait, entries = map(sum, zip(*lane_rows))
            if stopped + near + far == 0:
                scores.append(0.0)
                continue
            # Opposite approaches can discharge in parallel on this network.
            paired = min(row[0] + row[1] for row in lane_rows)
            age = min(4.0, (self.elapsed - self.last_green[phase]) / 30.0)
            features = [stopped, near, far, wait, paired, int(phase == current), age, entries]
            scores.append(self.weights @ features)
        return max(range(len(scores)), key=lambda phase: (scores[phase], phase == current))
