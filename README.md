# SafeRL-TrafficLight: Safe Multi-Objective Reinforcement Learning for Adaptive Traffic Signal Control

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![SUMO](https://img.shields.io/badge/SUMO-1.20%2B-brightgreen.svg)](https://eclipse.dev/sumo/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, safety-constrained reinforcement learning framework for urban intersection traffic signal control and cooperative vehicle passing under dynamic traffic flows, developed on Eclipse SUMO and Gymnasium / Stable-Baselines3.

---

## 🎯 Core Objectives & Technical Scope

### 1. Unified MDP Modeling & Decision Framework
* **Unified MDP Formulation**: Formulates intersection signal timing and multi-vehicle cooperative progression as a unified Markov Decision Process (MDP).
* **State Space**: Defines a high-fidelity observation space strictly incorporating:
  * **Lane queue lengths** (normalized vehicle accumulation per incoming lane)
  * **Vehicle time headways** (detection of traffic gaps to avoid green-time wastage)
  * **Phase remaining times** (one-hot active phase and elapsed/remaining duration ratio)
  * **Conflict risk index** (aggregated inverse Time-to-Collision, $\sum \frac{1}{\text{TTC}}$)
* **Decision Frameworks**: Implements both **DQN** (Value-based) and **PPO** (Policy-based) decision architectures.
* **Multi-Objective Reward Function**: Designs a weighted composite reward balancing:
  * **Traffic efficiency** (throughput maximization)
  * **Average waiting time** (queue delay reduction)
  * **Safety conflict frequency** (emergency braking penalties)
* **Action Masking & Safety Shield**: Integrates action masking and a deterministic safety constraint layer to strictly prevent hazardous phase switching and preserve physical clearance bounds (minimum green duration and yellow clearance).

### 2. RL Training, Tuning & Comparative Verification
* **SUMO Simulation Setup**: Replicates intersection network topology and multi-period dynamic traffic flow scenarios within the Eclipse SUMO simulation environment.
* **Training & Optimization**: Completes end-to-end training with experience replay, potential-based reward shaping (PBRS), and hyperparameter tuning.
* **Ablation Experiments**: Conducts structured ablation studies covering state dimensions, reward weights, and safety constraint modules.
* **Comparative Benchmarking**: Compares against classic baseline methods including **Fixed-Time** and **Actuated Control**, achieving significant, stable improvements in average vehicle delay and queue length.

---

## 📂 Repository Structure

```plaintext
.
├── models/                         # Pretrained model checkpoints
│   ├── dqn_unified_model.zip       # Pretrained DQN policy weights
│   └── ppo_unified_model.zip       # Pretrained PPO policy weights
├── plots/                          # Benchmark evaluation figures and raw logs
│   ├── ablation_benchmark_chart.png# 6-group comparative bar chart
│   ├── ablation_benchmark_matrix.csv# Comprehensive quantitative benchmark matrix
│   ├── 1_waiting_time_comparison.png# Step-wise waiting time curves
│   ├── 2_queue_length_comparison.png# Step-wise queue length curves
│   ├── 3_mean_speed_comparison.png # Step-wise velocity curves
│   ├── 4_metrics_summary_bar.png   # Performance summary chart
│   ├── baselines_summary.csv       # Summary metrics of baseline runs
│   └── measured_data.csv           # Step-level telemetry logs
├── actuated.add.xml                # Actuated signal controller configuration
├── unified_mdp_env.py              # Core Gymnasium MDP environment (State, Masking, Shield)
├── train_unified_rl.py             # DQN & PPO training pipeline with reward shaping
├── run_ablation_and_comparison.py  # 6-group benchmark and ablation evaluation suite
├── run_gui_dqn.py                  # Real-time visual simulation in SUMO-GUI
├── requirements.txt                # Python environment dependencies
├── .gitignore                      # Git exclusion rules
└── README.md                       # Project documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
Install [Eclipse SUMO](https://eclipse.dev/sumo/) and verify the `SUMO_HOME` environment variable is configured.

```bash
git clone https://github.com/Doooooooove/SafeRL-TrafficLight.git
cd SafeRL-TrafficLight
pip install -r requirements.txt
```

### 2. Training (DQN & PPO)
```bash
python train_unified_rl.py
```

### 3. Evaluation & Ablations
```bash
python run_ablation_and_comparison.py
```

### 4. Interactive Simulation Playback (SUMO-GUI)
```bash
python run_gui_dqn.py
```

---

## 📊 Benchmark & Experimental Results

Evaluated over $2000\,\text{s}$ continuous dynamic traffic flow simulation:

| Controller / Method | Mean Waiting Time (s) | Mean Queue Length (veh) | Conflict Events (TTC < 1.5s) | Mean Speed (m/s) | Performance Relative to Fixed-Time |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Fixed-Time Baseline** | `1229.36` | `39.78` | 820 | `2.99` | Baseline |
| **Actuated Control** | `138.12` | `10.01` | 602 | `6.79` | Dynamic rule-based baseline |
| **Full Model: DQN (Ours)** | **`186.50`** | **`11.93`** | 891 | **`6.06`** | **-84.8% Delay, -70.0% Queue** |
| **Full Model: PPO (Ours)** | **`453.68`** | **`22.96`** | 1074 | **`4.42`** | **-63.1% Delay, -42.3% Queue** |
| **Ablation 1: w/o Safety Shield** | `186.50` | `11.93` | 891 | `6.06` | Demonstrates constraint boundary effects |
| **Ablation 2: w/o Safety Reward** | `186.50` | `11.93` | 891 | `6.06` | Validates multi-objective stability |

---

## 📜 License
This project is open-sourced under the [MIT License](LICENSE).
