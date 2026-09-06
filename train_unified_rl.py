import os
import torch
from stable_baselines3 import DQN, PPO
from unified_mdp_env import UnifiedMDPIntersectionEnv

torch.set_num_threads(4)

def train_all_models(total_timesteps=4000):
    os.makedirs("D:/SUMO_RL_Project/models", exist_ok=True)

    # ================= 1. 训练 DQN 智能体 =================
    print("=" * 60, flush=True)
    print("    [1/2] 开始在统一 MDP 环境中训练 DQN 智能体 (均衡奖励)...", flush=True)
    print("=" * 60, flush=True)
    env_dqn = UnifiedMDPIntersectionEnv(
        num_seconds=20000,
        delta_time=5,
        min_green=5,
        w_efficiency=1.0,     # 放行车辆奖励
        w_delay=1.0,          # 延误削减奖励 (PBRS)
        w_conflict=2.0,       # 危险急刹车惩罚
        enable_safety_shield=True,
        sumo_seed=42,
    )
    dqn_model = DQN(
        policy="MlpPolicy",
        env=env_dqn,
        learning_rate=1e-3,
        buffer_size=20000,
        learning_starts=200,
        batch_size=64,
        gamma=0.95,
        train_freq=4,
        target_update_interval=250,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        verbose=1,
    )
    dqn_model.learn(total_timesteps=total_timesteps)
    dqn_path = "D:/SUMO_RL_Project/models/dqn_unified_model"
    dqn_model.save(dqn_path)
    env_dqn.close()
    print(f"[+] DQN 智能体训练完成，已保存至: {dqn_path}.zip\n", flush=True)

    # ================= 2. 训练 PPO 智能体 =================
    print("=" * 60, flush=True)
    print("    [2/2] 开始在统一 MDP 环境中训练 PPO 智能体 (PBRS)...", flush=True)
    print("=" * 60, flush=True)
    env_ppo = UnifiedMDPIntersectionEnv(
        num_seconds=20000,
        delta_time=5,
        min_green=5,
        w_efficiency=1.0,
        w_delay=1.0,
        w_conflict=2.0,
        enable_safety_shield=True,
        sumo_seed=42,
    )
    ppo_model = PPO(
        policy="MlpPolicy",
        env=env_ppo,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )
    ppo_model.learn(total_timesteps=total_timesteps)
    ppo_path = "D:/SUMO_RL_Project/models/ppo_unified_model"
    ppo_model.save(ppo_path)
    env_ppo.close()
    print(f"[+] PPO 智能体训练完成，已保存至: {ppo_path}.zip\n", flush=True)

if __name__ == "__main__":
    train_all_models(total_timesteps=6000)
