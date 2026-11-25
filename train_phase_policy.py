"""Fit the phase scorer on deterministic and generated traffic, using disjoint development seeds."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np

from phase_policy import FEATURES, MODEL_PATH, LearnedPhaseController


SCENARIO = Path(__file__).resolve().parent / "scenario"
ROUTES = (SCENARIO / "single-intersection-vhvh.rou.xml", SCENARIO / "alternating.rou.xml")
FIT_SEEDS = tuple(range(1800, 1808))
VALIDATION_SEEDS = (2901, 2902, 2903)
REFINEMENT_SEEDS = (7, 17, 29, 37)


def training_routes(directory, seed, duration):
    """Generate Poisson flows from the original network's route definitions."""
    if seed not in (*FIT_SEEDS, *VALIDATION_SEEDS, *REFINEMENT_SEEDS):
        raise ValueError(f"Seed {seed} is outside the development split")
    movements = ET.parse(ROUTES[0]).getroot().findall("route")
    rng = np.random.default_rng(seed)
    generated = []
    for profile in ("balanced", "varying"):
        root = ET.Element("routes")
        root.extend(movements)
        for begin in range(0, duration, 200):
            total_rate = rng.uniform(2000, 2650 if profile == "balanced" else 2800)
            approach_weights = rng.uniform(0.70, 1.30, 4)
            if profile == "varying":
                approach_weights[rng.choice(4, size=2, replace=False)] *= 0.55
            weights = {}
            for movement in movements:
                source, target = movement.attrib["edges"].split()
                approach = "nesw".index(source[0])
                turn = ("nesw".index(target[-1]) - approach) % 4
                weights[movement.attrib["id"]] = approach_weights[approach] * (
                    (1.0, 1.0, 1.6, 0.55)[turn] * rng.uniform(0.6, 1.4)
                )
            total_weight = sum(weights.values())
            for route, weight in weights.items():
                rate = total_rate * weight / total_weight / 3600
                ET.SubElement(root, "flow", id=f"{route}_{begin}", route=route, begin=str(begin),
                              end=str(min(begin + 200, duration)), period=f"exp({rate:.8f})",
                              departSpeed="max", departPos="base", departLane="best")
        path = directory / f"{profile}-{seed}.rou.xml"
        ET.ElementTree(root).write(path, encoding="utf-8")
        generated.append(path)
    return (*ROUTES, *generated)


def episode_cost(weights, route, seed, duration):
    from run_ablation_and_comparison import INTERVAL, _trip_metrics
    from unified_mdp_env import _CountingEnvironment, _SumoArguments

    with TemporaryDirectory(prefix="phase-trips-") as directory:
        trips = Path(directory) / "trips.xml"
        env = _CountingEnvironment(
            net_file=str(SCENARIO / "single-intersection.net.xml"), route_file=str(route),
            num_seconds=duration, single_agent=True, delta_time=INTERVAL, sumo_seed=seed,
            additional_sumo_cmd=_SumoArguments([
                "--no-step-log", "--tripinfo-output", str(trips),
                "--tripinfo-output.write-unfinished", "--tripinfo-output.write-undeparted",
            ]),
            add_system_info=False, add_per_agent_info=False, reward_fn=lambda signal: 0.0,
        )
        try:
            env.reset(seed=seed)
            signal = env.traffic_signals[env.ts_ids[0]]
            policy = LearnedPhaseController(weights)
            vehicle = signal.sumo.vehicle
            stopped = 0
            decisions = duration // INTERVAL
            for _ in range(decisions):
                can_switch = signal.time_since_last_phase_change >= signal.yellow_time + signal.min_green
                env.step(policy.choose_phase(signal, can_switch))
                stopped += sum(vehicle.getSpeed(v) < 0.1 for v in vehicle.getIDList())
        finally:
            env.close()
        arrivals = env.interval_arrivals
        trip_metrics = _trip_metrics(trips, arrivals)
    scheduled = trip_metrics["scheduled_veh"]
    unfinished = (scheduled - arrivals) / scheduled
    return stopped / decisions, trip_metrics["demand_mean_delay_s"], unfinished


def matched_cost(task):
    weights, routes, seed, duration = task
    return [episode_cost(weights, route, seed, duration) for route in routes]


def rank_cost(metrics):
    # Emphasize queues on the two deterministic profiles and accrued delay on
    # generated traffic. Unfinished trips include vehicles waiting to enter.
    queue, delay, unfinished = np.moveaxis(metrics, -1, 0)
    costs = queue + delay * (0.10, 0.10, 1.10, 0.75) + unfinished * (30, 30, 280, 180)
    relative = costs / np.median(costs, axis=0)
    return relative @ (0.37, 0.37, 0.18, 0.08) + 0.10 * relative[:, :2].max(axis=1)


def fit(workers):
    rng = np.random.default_rng(86)
    center = np.array([1.0, 1.5, 0.0, 0.4, 0.4, 2.0, 0.8, -0.2, 0.8, 1.6, -0.3, 0.2])
    spread = np.array([0.0, 0.9, 0.4, 0.5, 1.0, 2.0, 0.65, 0.6, 1.0, 1.2, 0.6, 1.0])
    minimum_spread = 0.20 * spread
    retained = []
    leaders = [center]
    with TemporaryDirectory(prefix="phase-demand-") as directory, ProcessPoolExecutor(max_workers=workers) as pool:
        for generation, seed in enumerate(FIT_SEEDS, start=1):
            duration = 1200
            routes = training_routes(Path(directory), seed, duration)
            group = np.clip(rng.normal(center, spread, size=(72, len(FEATURES))), -1.5, 8.0)
            group[:, 0] = 1.0  # Fix the units of the score at one stopped vehicle.
            group[0] = center
            group[1:1 + len(leaders)] = leaders
            metrics = np.asarray(list(pool.map(matched_cost, [(w, routes, seed, duration) for w in group])))
            order = np.argsort(rank_cost(metrics), kind="stable")
            leaders = group[order[:8]]
            center = leaders.mean(axis=0)
            spread = np.maximum(leaders.std(axis=0) * 1.25, minimum_spread)
            retained.extend(leaders[:5])
            print(f"Generation {generation}/{len(FIT_SEEDS)}, best mean queue "
                  f"{metrics[order[0], :, 0].mean():.2f}", flush=True)

        weights = np.unique([*retained, center], axis=0)
        print("Validating shortlisted policies on full episodes", flush=True)
        metrics = full_episode_metrics(pool, directory, weights, VALIDATION_SEEDS)
        top = weights[np.argsort(rank_cost(metrics), kind="stable")[:6]]

        # Refine the best full-episode policies using smaller perturbations and
        # additional development seeds to reduce sensitivity to a single rollout.
        nearby = rng.normal(top[:, None, :], 1.5 * minimum_spread, size=(len(top), 10, len(FEATURES)))
        weights = np.concatenate([top, np.clip(nearby.reshape(-1, len(FEATURES)), -1.5, 8.0)])
        print("Scoring refined policies on additional development seeds", flush=True)
        metrics = full_episode_metrics(pool, directory, weights, REFINEMENT_SEEDS)
    winner = np.argmin(rank_cost(metrics))
    print("Development metrics by profile (queue, accrued delay, unfinished fraction)", metrics[winner].tolist())
    return weights[winner]


def full_episode_metrics(pool, directory, weights, seeds):
    duration = 2000
    profiles = [(training_routes(Path(directory), seed, duration), seed) for seed in seeds]
    tasks = [(w, routes, seed, duration) for w in weights for routes, seed in profiles]
    return np.asarray(list(pool.map(matched_cost, tasks))).reshape(
        len(weights), len(seeds), len(ROUTES) + 2, 3
    ).mean(axis=1)


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
