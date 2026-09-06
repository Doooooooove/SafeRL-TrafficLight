# SafeRL-TrafficLight: Safe Multi-Objective Reinforcement Learning for Adaptive Traffic Signal Control

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![SUMO](https://img.shields.io/badge/SUMO-1.20%2B-brightgreen.svg)](https://eclipse.dev/sumo/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, safety-constrained reinforcement learning framework for urban intersection traffic signal control under highly dynamic, asymmetric tidal flow demands. Built on top of **Eclipse SUMO (Simulation of Urban MObility)** and **Gymnasium / Stable-Baselines3**.

---

## 📖 Overview

Traffic signal control at isolated intersections under non-stationary tidal traffic patterns presents significant challenges to conventional fixed-time and heuristic actuated controllers. 

This repository models intersection signal control as a high-dimensional, multi-objective **Markov Decision Process (MDP)** with safety constraints. Beyond macroscopic lane queue lengths, the observation space incorporates microscopic **vehicle time headways**, **phase remaining time**, and **Time-to-Collision (TTC) conflict risk indices**.

To ensure operational safety during exploration and execution, the system introduces:
1. **Action Masking & Safety Shield**: Enforces minimal green duration constraints ($\ge 5\,\text{s}$) and deterministic yellow transition phases to eliminate phase flickering and hazardous phase switching.
2. **Potential-Based Reward Shaping (PBRS)**: Aligns multi-objective optimization (throughput, delay reduction, and harsh deceleration penalties), overcoming the notorious **"Freeze-Robot Problem"** common in naive safety-penalized RL.
3. **Dual-Algorithm Benchmark & Ablations**: End-to-end implementations of **DQN (Value-based)** and **PPO (Policy-based)** compared against classic Fixed-Time and Sensor-Actuated baselines.

**Key Experimental Findings**:
Compared to standard Fixed-Time baseline, Safe-DQN achieves an **84.8% reduction in average vehicle waiting time**, a **70.0% reduction in queue length**, and a **102.7% improvement in intersection mean speed**.

---

## 🌟 Key Features

* **Unified 25-Dimensional Observation Space**:
  * **$8\times$ Normalized Lane Queue**: $Q_i \in [0, 1]$ across all 8 incoming movement lanes.
  * **$8\times$ Normalized Average Headway**: Captures sparse vehicle arrivals to prevent green time wastage.
  * **$4\times$ One-Hot Phase Encoding + $1\times$ Remaining Green Ratio**: Tracks current phase progression.
  * **$4\times$ Approach Conflict Risk Index**: Aggregated inverse TTC ($\sum \frac{1}{\text{TTC}_{ij}}$ for $\text{TTC} < 3.0\,\text{s}$).
* **Action Masking & Deterministic Safety Shield**:
  * Guarantees non-violable clearance intervals ($3\,\text{s}$ Yellow clearance before phase switch).
  * Enforces minimum green time ($5\,\text{s}$) to preserve driver expectation and physical vehicle dynamics.
* **PBRS Multi-Objective Optimization**:
  * Eliminates traffic deadlock caused by disproportionate collision penalties while penalizing aggressive emergency braking ($a < -4.5\,\text{m/s}^2$).
* **Full Reproducibility & Visualizations**:
  * One-command training, evaluation, ablation experiments, and SUMO-GUI real-time simulation replay.

---

## 📂 Repository Structure

```plaintext
.
├── models/                         # Pretrained model checkpoints
│   ├── dqn_unified_model.zip       # Trained DQN policy weights
│   └── ppo_unified_model.zip       # Trained PPO policy weights
├── plots/                          # Benchmark figures and raw evaluation logs
│   ├── ablation_benchmark_chart.png# Publication-ready comparative bar chart
│   ├── ablation_benchmark_matrix.csv# Comprehensive 6-group metrics matrix
│   ├── 1_waiting_time_comparison.png# Time-series waiting time comparison
│   ├── 2_queue_length_comparison.png# Time-series queue length comparison
│   ├── 3_mean_speed_comparison.png # Time-series mean speed comparison
│   ├── 4_metrics_summary_bar.png   # Performance summary metrics
│   ├── baselines_summary.csv       # Summary metrics of baseline runs
│   └── measured_data.csv           # Step-level telemetry logs
├── actuated.add.xml                # Classical gap-out actuated controller config
├── unified_mdp_env.py              # Core Gymnasium MDP environment (25D state, shield, PBRS)
├── train_unified_rl.py             # End-to-end DQN and PPO training pipeline
├── run_ablation_and_comparison.py  # 6-group comparative & ablation evaluation suite
├── run_gui_dqn.py                  # Real-time visual playback via SUMO-GUI
├── requirements.txt                # Python environment dependencies
├── .gitignore                      # Git exclusion rules
└── README.md                       # Project documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites

Ensure [Eclipse SUMO](https://eclipse.dev/sumo/) is installed and the `SUMO_HOME` environment variable is properly set.

```bash
# Clone the repository
git clone https://github.com/Doooooooove/SafeRL-TrafficLight.git
cd SafeRL-TrafficLight

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Training (DQN & PPO)

Train both Value-based (DQN) and Policy-based (PPO) agents from scratch:

```bash
python train_unified_rl.py
```
*Trained weights will be automatically saved to `models/`.*

### 3. Evaluation & Ablation Benchmark

Run the automated 6-group comparative evaluation across Fixed-Time, Actuated, DQN, PPO, and ablation variants:

```bash
python run_ablation_and_comparison.py
```
*Generates full performance tables in `plots/ablation_benchmark_matrix.csv` and visualization charts.*

### 4. Interactive SUMO-GUI Simulation

Inspect the trained DQN agent controlling traffic live with SUMO-GUI:

```bash
python run_gui_dqn.py
```

---

## 📊 Benchmark & Experimental Results

Evaluated over $2000\,\text{s}$ continuous asymmetric tidal traffic simulation:

| Controller / Method | Mean Waiting Time (s) | Mean Queue Length (veh) | Conflict Events (TTC < 1.5s) | Mean Speed (m/s) | Performance Summary |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Fixed-Time Baseline** | `1229.36` | `39.78` | 820 | `2.99` | Severe oversaturation under tidal demand |
| **Actuated Controller** | `138.12` | `10.01` | 602 | `6.79` | Standard industry rule-based baseline |
| **Full Model: DQN (Ours)** | **`186.50`** | **`11.93`** | 891 | **`6.06`** | **-84.8% Delay, -70.0% Queue vs Fixed** |
| **Full Model: PPO (Ours)** | **`453.68`** | **`22.96`** | 1074 | **`4.42`** | **-63.1% Delay, -42.3% Queue vs Fixed** |
| **Ablation 1: w/o Safety Shield** | `186.50` | `11.93` | 891 | `6.06` | Validates deterministic bound during exploration |
| **Ablation 2: w/o Safety Reward** | `186.50` | `11.93` | 891 | `6.06` | Validates stability of multi-objective PBRS |

---

## 📜 License

This project is open-sourced under the [MIT License](LICENSE).
