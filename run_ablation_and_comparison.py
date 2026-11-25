"""Compare traffic-signal controllers on matched SUMO episodes."""

import argparse
from pathlib import Path

import pandas as pd

from queue_controller import QueueController

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ROUTE = PROJECT_DIR / "scenario" / "single-intersection-vhvh.rou.xml"
CONTROLLERS = ("fixed", "actuated", "dqn", "ppo", "adaptive-dqn", "queue")
LABELS = {"fixed": "Fixed-time", "actuated": "Actuated", "dqn": "DQN",
          "ppo": "PPO", "adaptive-dqn": "Adaptive DQN", "queue": "Queue-aware"}
METRICS = ("mean_queue_veh", "arrivals_veh", "mean_wait_s", "mean_speed_mps",
           "ttc_pair_samples", "hard_brake_samples")
INTERVAL = 5


def evaluate_episode(controller, model, seed, seconds, begin, route_file):
    from unified_mdp_env import UnifiedMDPIntersectionEnv

    native = controller in ("fixed", "actuated")
    program_id = {"fixed": "0", "actuated": "actuated"}.get(controller)
    additional = ["--no-step-log"]
    if controller == "actuated":
        additional.extend(["-a", str(PROJECT_DIR / "actuated.add.xml")])
    env = UnifiedMDPIntersectionEnv(
        fixed_ts=native, program_id=program_id, additional_sumo_cmd=additional,
        sumo_seed=seed, num_seconds=seconds, begin_time=begin,
        delta_time=INTERVAL, route_file=route_file,
    )
    queue_policy = QueueController() if controller == "queue" else None
    stopped = waiting = speed = 0.0
    arrivals = near_pairs = hard_brakes = 0
    try:
        obs, _ = env.reset(seed=seed)
        if native:
            from traci import constants as tc

            tls = env.env.sumo.trafficlight
            active = tls.getProgram(env.ts_id)
            logics = {logic.programID: logic.type for logic in tls.getAllProgramLogics(env.ts_id)}
            expected = tc.TRAFFICLIGHT_TYPE_STATIC if controller == "fixed" else tc.TRAFFICLIGHT_TYPE_ACTUATED
            if active != program_id or logics.get(active) != expected:
                raise RuntimeError(f"Expected native {controller} program, got {active!r} ({logics}).")

        for step in range(1, seconds // INTERVAL + 1):
            if queue_policy is not None:
                action = queue_policy.predict(env)
            elif model is not None:
                action = int(model.predict(obs, deterministic=True)[0].item())
            else:
                action = 0  # Ignored by the native signal program.
            obs, _, terminated, truncated, info = env.step(action)
            stopped += info["system_total_stopped"]
            waiting += info["system_mean_waiting_time"]
            speed += info["system_mean_speed"]
            arrivals += info["step_throughput"]
            near_pairs += info["conflict_count"]
            hard_brakes += info["hard_brakes"]
            if terminated or truncated:
                break
        if info["step"] != begin + seconds:
            raise RuntimeError(f"Expected end at {begin + seconds}s, reached {info['step']}s.")
    finally:
        env.close()
    return {
        "controller": controller, "route": route_file.name, "seed": seed,
        "begin_s": begin, "duration_s": seconds,
        "mean_queue_veh": stopped / step, "arrivals_veh": arrivals,
        "mean_wait_s": waiting / step, "mean_speed_mps": speed / step,
        "ttc_pair_samples": near_pairs, "hard_brake_samples": hard_brakes,
    }


def save_results(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(output_dir / "runs.csv", index=False)
    groups = runs.groupby(["controller", "route", "begin_s", "duration_s"], sort=False)
    summary = groups[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{'sd' if stat == 'std' else stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_seeds", groups.size())
    summary = summary.reset_index()
    summary.to_csv(output_dir / "summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, layout="constrained")
    for ax, metric, xlabel in zip(axes, ("mean_queue_veh", "arrivals_veh"),
                                  ("Mean stopped vehicles", "Vehicles arriving within horizon")):
        for row, controller in enumerate(summary["controller"]):
            mean = summary.loc[row, f"{metric}_mean"]
            sd = summary.loc[row, f"{metric}_sd"]
            color = "#087e80" if controller in ("queue", "adaptive-dqn") else "#47566a"
            ax.errorbar(mean, row, xerr=sd if pd.notna(sd) else None, fmt="o",
                        color=color, capsize=3, markersize=6)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(summary)), summary["controller"].map(LABELS))
    axes[0].invert_yaxis()
    fig.savefig(output_dir / "comparison.png", dpi=170)
    plt.close(fig)
    print(summary[["controller", "n_seeds", "mean_queue_veh_mean", "arrivals_veh_mean"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--seconds", type=int, default=2000, help="Episode length in seconds (multiple of 5).")
    parser.add_argument("--begin", type=int, default=0, help="SUMO start time in seconds.")
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=CONTROLLERS)
    parser.add_argument("--route-file", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results")
    args = parser.parse_args()
    if args.seconds <= 0 or args.seconds % INTERVAL or args.begin < 0:
        parser.error("--seconds must be a positive multiple of 5 and --begin nonnegative.")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("--seeds must be distinct, nonnegative integers.")
    if len(set(args.controllers)) != len(args.controllers):
        parser.error("Each controller may only appear once.")
    route_file = args.route_file.resolve()
    if not route_file.is_file():
        parser.error(f"Route file does not exist: {route_file}")

    rows = []
    for controller in args.controllers:
        model = None
        if controller in ("dqn", "ppo", "adaptive-dqn"):
            from stable_baselines3 import DQN, PPO

            algorithm = PPO if controller == "ppo" else DQN
            filename = "adaptive_dqn.zip" if controller == "adaptive-dqn" else f"{controller}_unified_model.zip"
            model = algorithm.load(PROJECT_DIR / "models" / filename, device="cpu")
        for seed in args.seeds:
            print(f"{LABELS[controller]} | seed {seed} | {args.seconds} s", flush=True)
            rows.append(evaluate_episode(controller, model, seed, args.seconds, args.begin, route_file))
    save_results(rows, args.output_dir)


if __name__ == "__main__":
    main()
