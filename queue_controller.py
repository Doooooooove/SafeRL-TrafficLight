"""Demand-responsive signal control for the four-phase intersection."""


class QueueController:
    def __init__(self):
        self.phase_lanes = None

    def predict(self, env):
        signal = env.env.traffic_signals[env.ts_id]
        current = signal.green_phase
        if not env.get_action_mask().all():
            return current

        if self.phase_lanes is None:
            links = signal.sumo.trafficlight.getControlledLinks(signal.id)
            self.phase_lanes = [
                {link[0] for color, group in zip(phase.state, links)
                 if color in "gG" for link in group}
                for phase in signal.green_phases
            ]

        # Moving arrivals can use an imminent green. Waiting vehicles receive
        # progressively more weight so quieter approaches also get service.
        lane_api = signal.sumo.lane
        demand = {
            lane: lane_api.getLastStepVehicleNumber(lane)
            + lane_api.getWaitingTime(lane) / 30.0
            for lane in signal.lanes
        }
        scores = [sum(demand[lane] for lane in lanes) for lanes in self.phase_lanes]
        # Keep useful green time when the competing demand barely differs.
        if scores[current] > 0:
            scores[current] += 2.0
        return max(range(len(scores)), key=lambda i: (scores[i], i == current))
