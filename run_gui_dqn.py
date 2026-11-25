"""Watch a controller operate the bundled intersection in SUMO-GUI."""

import argparse
from pathlib import Path

from controllers import PLAYBACK_CONTROLLERS, episode_policy
from unified_mdp_env import UnifiedMDPIntersectionEnv


def run_gui_dqn(num_seconds=1000, controller="dqn", seed=42, use_gui=True, route_file=None):
    policy = episode_policy(controller)
    env = UnifiedMDPIntersectionEnv(
        use_gui=use_gui, num_seconds=num_seconds, sumo_seed=seed, route_file=route_file,
        additional_sumo_cmd="--delay 60" if use_gui else None,
    )
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            action = policy(env, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=PLAYBACK_CONTROLLERS, default="dqn")
    parser.add_argument("--seconds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--route-file", type=Path, help="Override the bundled traffic demand.")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    run_gui_dqn(args.seconds, args.controller, args.seed, route_file=args.route_file)
