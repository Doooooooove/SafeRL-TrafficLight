"""Train DQN and PPO on the same SUMO observation and reward."""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import DQN, PPO

from unified_mdp_env import UnifiedMDPIntersectionEnv


MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_TYPES = {"dqn": DQN, "ppo": PPO}


def train_all_models(total_timesteps=4000, algorithms=("dqn", "ppo"), seed=42, output_dir=MODEL_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    for name in algorithms:
        env = UnifiedMDPIntersectionEnv(
            num_seconds=20000, w_efficiency=1.0, w_delay=1.0, w_conflict=2.0, sumo_seed=seed,
        )
        if name == "dqn":
            options = dict(learning_rate=1e-3, buffer_size=20000, learning_starts=200,
                           batch_size=64, train_freq=4, target_update_interval=250,
                           exploration_fraction=0.35, exploration_final_eps=0.02)
        else:
            options = dict(learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10,
                           clip_range=0.2, ent_coef=0.01)
        try:
            model = MODEL_TYPES[name]("MlpPolicy", env, gamma=0.95, seed=seed,
                                     device="cpu", verbose=1, **options)
            model.learn(total_timesteps=total_timesteps)
            model.save(output_dir / f"{name}_unified_model.zip")
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("dqn", "ppo", "both"), default="both")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    algorithms = ("dqn", "ppo") if args.algorithm == "both" else (args.algorithm,)
    train_all_models(args.steps, algorithms, args.seed, args.output_dir)
