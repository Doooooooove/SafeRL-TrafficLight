import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from unified_mdp_env import UnifiedMDPIntersectionEnv

def evaluate_model(env, model, num_steps=400):
    obs, info = env.reset(seed=42)
    done = False
    waits, queues, conflicts, spds, throughs = [], [], [], [], []
    step = 0
    while not done and step < num_steps:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = 0 # baseline fixed
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        waits.append(info["system_waiting_time"])
        queues.append(info.get("system_total_stopped", 0))
        conflicts.append(info["conflict_count"])
        spds.append(info.get("system_mean_speed", 0))
        throughs.append(info["step_throughput"])
    env.close()
    return {
        "avg_waiting": np.mean(waits),
        "avg_queue": np.mean(queues),
        "total_conflicts": int(np.sum(conflicts)),
        "avg_speed": np.mean(spds),
        "total_throughput": int(np.sum(throughs)),
    }

def run_all_evaluations():
    print("=" * 70)
    print("        【全量算法对比与消融实验 (Benchmark & Ablation Study)】")
    print("=" * 70)

    dqn_path = "D:/SUMO_RL_Project/models/dqn_unified_model.zip"
    ppo_path = "D:/SUMO_RL_Project/models/ppo_unified_model.zip"

    results = []

    # 1. 传统定时基线 (Fixed-Time)
    print("[1/6] 评测基线: 传统定时配时 (Fixed-Time)...")
    env_fixed = UnifiedMDPIntersectionEnv(num_seconds=2000, sumo_seed=42)
    env_fixed.env.fixed_ts = True
    r_fixed = evaluate_model(env_fixed, model=None)
    r_fixed["model_name"] = "基线: 定时配时 (Fixed-Time)"
    results.append(r_fixed)

    # 2. 传统动态感应控制 (Actuated)
    print("[2/6] 评测基线: 动态感应控制 (Actuated Control)...")
    env_act = UnifiedMDPIntersectionEnv(
        num_seconds=2000, sumo_seed=42, additional_sumo_cmd="-a D:/SUMO_RL_Project/actuated.add.xml"
    )
    env_act.env.fixed_ts = True
    env_act.reset(seed=42)
    ts = env_act.env.unwrapped.traffic_signals[env_act.ts_id]
    ts.sumo.trafficlight.setProgram(ts.id, "actuated")
    r_act = evaluate_model(env_act, model=None)
    r_act["model_name"] = "基线: 感应控制 (Actuated)"
    results.append(r_act)

    # 3. DQN 统一智能体 (完整模型)
    print("[3/6] 评测算法: 完整 DQN 智能体 (Ours)...")
    dqn_model = DQN.load(dqn_path)
    env_dqn = UnifiedMDPIntersectionEnv(num_seconds=2000, sumo_seed=42)
    r_dqn = evaluate_model(env_dqn, dqn_model)
    r_dqn["model_name"] = "完整模型: DQN (Ours)"
    results.append(r_dqn)

    # 4. PPO 统一智能体 (完整模型)
    print("[4/6] 评测算法: 完整 PPO 智能体 (Ours)...")
    ppo_model = PPO.load(ppo_path)
    env_ppo = UnifiedMDPIntersectionEnv(num_seconds=2000, sumo_seed=42)
    r_ppo = evaluate_model(env_ppo, ppo_model)
    r_ppo["model_name"] = "完整模型: PPO (Ours)"
    results.append(r_ppo)

    # 5. 消融实验 1: 移除安全约束层 (w/o Safety Shield)
    print("[5/6] 消融实验 1: 移除安全约束层 (w/o Safety Shield)...")
    env_no_shield = UnifiedMDPIntersectionEnv(num_seconds=2000, enable_safety_shield=False, sumo_seed=42)
    r_no_shield = evaluate_model(env_no_shield, dqn_model)
    r_no_shield["model_name"] = "消融: 无安全约束层 (w/o Shield)"
    results.append(r_no_shield)

    # 6. 消融实验 2: 奖励函数移除安全冲突惩罚 (w/o Safety Reward)
    print("[6/6] 消融实验 2: 移除安全冲突惩罚 (w3=0, w/o Safety Reward)...")
    env_no_sec_rew = UnifiedMDPIntersectionEnv(num_seconds=2000, w_conflict=0.0, sumo_seed=42)
    r_no_sec_rew = evaluate_model(env_no_sec_rew, dqn_model)
    r_no_sec_rew["model_name"] = "消融: 移除安全惩罚 (w3=0)"
    results.append(r_no_sec_rew)

    # 构造 DataFrame
    df = pd.DataFrame(results)
    cols = ["model_name", "avg_waiting", "avg_queue", "total_conflicts", "avg_speed", "total_throughput"]
    df = df[cols]
    csv_out = "D:/SUMO_RL_Project/plots/ablation_benchmark_matrix.csv"
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 85)
    print("                    【全量算法对比与消融实验成果矩阵】")
    print("=" * 85)
    print(f"{'模型/实验组':<30} | {'平均等待(s)':<12} | {'平均排队(辆)':<12} | {'安全冲突(次)':<12} | {'车速(m/s)':<10}")
    print("-" * 85)
    for _, row in df.iterrows():
        print(f"{row['model_name']:<30} | {row['avg_waiting']:<12.2f} | {row['avg_queue']:<12.2f} | {row['total_conflicts']:<12} | {row['avg_speed']:<10.2f}")
    print("=" * 85)

    # 绘制高精度消融与对比柱状图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    names = ["定时配时", "感应控制", "DQN(全)", "PPO(全)", "无安全层", "无安全奖"]
    colors = ['#ff7675', '#fdcb6e', '#00b894', '#0984e3', '#e17055', '#d63031']

    # 1. 等待时间
    axes[0].bar(names, df['avg_waiting'], color=colors, edgecolor='black')
    axes[0].set_title("平均车辆等待时间 (s)", fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].grid(axis='y', linestyle=':', alpha=0.7)

    # 2. 排队长度
    axes[1].bar(names, df['avg_queue'], color=colors, edgecolor='black')
    axes[1].set_title("平均排队停顿车数 (辆)", fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].grid(axis='y', linestyle=':', alpha=0.7)

    # 3. 安全冲突次数
    axes[2].bar(names, df['total_conflicts'], color=colors, edgecolor='black')
    axes[2].set_title("安全冲突总次数 (TTC<1.5s)", fontsize=12, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=30)
    axes[2].grid(axis='y', linestyle=':', alpha=0.7)

    plt.suptitle("全维度算法对比与消融实验分析 (Benchmark & Ablation Study)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_out = "D:/SUMO_RL_Project/plots/ablation_benchmark_chart.png"
    plt.savefig(plot_out)
    plt.close()
    print(f"[+] 综合对比与消融图表已生成至: {plot_out}")

if __name__ == "__main__":
    run_all_evaluations()
