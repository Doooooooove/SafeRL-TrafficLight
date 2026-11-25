"""Fit the phase scorer by cross-entropy policy search on two SUMO demand profiles."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from sumo_rl import SumoEnvironment

from phase_policy import FEATURES, MODEL_PATH, LearnedPhaseController


SCENARIO = Path(__file__).resolve().parent / "scenario"
ROUTES = ("single-intersection-vhvh.rou.xml", "alternating.rou.xml")


def episode_cost(weights, route, seed, duration):
    env = SumoEnvironment(
        net_file=str(SCENARIO / "single-intersection.net.xml"), route_file=str(SCENARIO / route),
        num_seconds=duration, single_agent=True, sumo_seed=seed, additional_sumo_cmd="--no-step-log",
        add_system_info=False, add_per_agent_info=False, reward_fn=lambda signal: 0.0,
    )
    try:
        env.reset(seed=seed)
        signal = env.traffic_signals[env.ts_ids[0]]
        policy = LearnedPhaseController(weights)
        queue = wait = 0.0
        decisions = duration // env.delta_time
        for _ in range(decisions):
            can_switch = signal.time_since_last_phase_change >= signal.yellow_time + signal.min_green
            env.step(policy.choose_phase(signal, can_switch))
            queue += sum(signal.sumo.lane.getLastStepHaltingNumber(lane) for lane in signal.lanes)
            lane_wait = sum(signal.sumo.lane.getWaitingTime(lane) for lane in signal.lanes)
            wait += lane_wait / max(1, signal.sumo.vehicle.getIDCount())
        return (queue + 0.5 * wait) / decisions
    finally:
        env.close()


def matched_cost(task):
    weights, seed, duration = task
    return np.mean([episode_cost(weights, route, seed, duration) for route in ROUTES])


def fit(workers):
    rng = np.random.default_rng(86)
    center = np.array([1.0, 1.0, 1.0, 1.0, 0.3, 2.0, 1.0, 1.0])
    spread = np.array([0.0, 0.7, 0.7, 0.9, 1.2, 1.5, 1.2, 1.4])
    minimum_spread = np.array([0.0, 0.15, 0.15, 0.12, 0.25, 0.25, 0.25, 0.25])
    retained = []
    leaders = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for generation in range(6):
            group = np.maximum(-0.2, rng.normal(center, spread, size=(28, len(center))))
            # Anchor the stopped-vehicle coefficient to fix the units of the score.
            group[:, 0] = 1.0
            group[0] = center
            group[1] = [1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0]
            for i, previous in enumerate(leaders[:2]):
                group[2 + i] = previous
            costs = list(pool.map(matched_cost, [(w, 1800 + generation, 1000) for w in group]))
            leaders = group[np.argsort(costs, kind="stable")[:6]]
            center = leaders.mean(axis=0)
            spread = np.maximum(leaders.std(axis=0) * 1.2, minimum_spread)
            retained.extend(leaders[:3])
        # Re-rank shortlisted policies over full episodes on common, separate seeds.
        weights = np.unique(retained, axis=0)
        tasks = [(w, seed, 2000) for w in weights for seed in (2901, 2902, 2903)]
        scores = np.asarray(list(pool.map(matched_cost, tasks))).reshape(len(weights), 3)
    return weights[np.argmin(scores.mean(axis=1))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="Parallel SUMO processes.")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    weights = fit(args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(zip(FEATURES, weights)), indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
