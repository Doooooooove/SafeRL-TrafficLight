"""Train the adaptive DQN on the original and alternating SUMO traffic patterns."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN

from queue_controller import QueueController
from unified_mdp_env import UnifiedMDPIntersectionEnv


ROOT = Path(__file__).resolve().parent
ROUTES = ("single-intersection-vhvh.rou.xml", "alternating.rou.xml")


def train(seed, output):
    rng = np.random.default_rng(seed)
    torch.set_num_threads(1)
    transitions = []
    model = None
    for episode in range(28):
        route = ROOT / "scenario" / ROUTES[episode % len(ROUTES)]
        env = UnifiedMDPIntersectionEnv(
            route_file=route, num_seconds=2000, additional_sumo_cmd=["--no-step-log"]
        )
        try:
            if model is None:
                model = DQN(
                    "MlpPolicy", env, seed=seed, gamma=0.92, learning_rate=0.001,
                    device="cpu", buffer_size=11200, policy_kwargs={"net_arch": [128, 128]},
                )
            obs, _ = env.reset(seed=10000 + seed + episode)
            teacher = QueueController()
            for _ in range(400):
                expert_action = teacher.predict(env)
                action = expert_action
                if env.get_action_mask().all() and rng.random() < 0.20:
                    action = int(rng.integers(env.action_space.n))
                next_obs, _, terminated, truncated, info = env.step(action)
                reward = (0.12 * info["step_throughput"] - 0.08 * info["system_total_stopped"]
                          - 0.20 * info["hard_brakes"])
                transitions.append((obs, action, reward, next_obs, info["action_mask"],
                                    terminated, expert_action))
                obs = next_obs
                if terminated or truncated:
                    break
        finally:
            env.close()

    obs, actions, rewards, next_obs, masks, terminal, expert = (
        torch.as_tensor(np.asarray(column)) for column in zip(*transitions)
    )
    for update in range(3600):
        batch = torch.from_numpy(rng.integers(len(obs), size=128))
        q_values = model.q_net(obs[batch])
        q_action = q_values.gather(1, actions[batch, None]).squeeze(1)
        with torch.no_grad():
            # Double DQN selects a legal next phase online and scores it with the target network.
            next_q = model.q_net(next_obs[batch]).masked_fill(~masks[batch], -torch.inf)
            best = next_q.argmax(dim=1, keepdim=True)
            bootstrap = model.q_net_target(next_obs[batch]).gather(1, best).squeeze(1)
            targets = rewards[batch].float() + model.gamma * (~terminal[batch]) * bootstrap
        loss = F.smooth_l1_loss(q_action, targets) + 2.0 * F.cross_entropy(q_values, expert[batch])
        model.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.q_net.parameters(), 10.0)
        model.policy.optimizer.step()
        if (update + 1) % 100 == 0:
            model.q_net_target.load_state_dict(model.q_net.state_dict())
    model.num_timesteps = len(transitions)
    model._n_updates = update + 1
    model.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "adaptive_dqn.zip")
    args = parser.parse_args()
    train(args.seed, args.output)
