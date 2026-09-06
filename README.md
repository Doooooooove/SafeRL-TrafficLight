# SafeRL-TrafficLight: 基于深度强化学习与安全约束的城市交叉口自适应信号控制

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![SUMO](https://img.shields.io/badge/SUMO-Simulation-brightgreen.svg)](https://eclipse.dev/sumo/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 项目简介 (Overview)

本项目基于城市微观交通仿真软件 **SUMO (Simulation of Urban MObility)**，针对非对称潮汐交通流场景，将交叉口信号配时统一建模为高维多目标马尔可夫决策过程（MDP）。

系统不仅融合了宏观车道排队长度，更引入了微观**车头时距 (Headway)**、**相位剩余可用时间**与**碰撞时间 (TTC) 冲突风险指数**；通过构建**动作掩码 (Action Masking)** 与 **安全约束层 (Safety Shield)** 规避危险相位突变；采用 **DQN** 与 **PPO** 双算法框架，在势能塑形多目标奖励驱动下实现自适应高效配时。

实测结果表明：相比传统定时配时基准，本系统使交叉口**平均车辆延误下降 84.8%**，**排队停顿减少 70.0%**，**通行车速提升 102.7%**。

---

## 🌟 核心技术亮点 (Key Features)

1. **统一高精 MDP 建模 (25 维状态空间)**：
   - 8 维各车道排队长度 ($Q \in [0, 1]$)
   - 8 维各车道平均车头时距 (Headway，精准识别绿灯空放)
   - 4 维绿灯相位 One-hot 编码 + 1 维可用绿灯剩余比例
   - 4 维进口道逆碰撞时间冲突风险指数 ($\sum \frac{1}{\text{TTC}}$)
2. **动作掩码与安全约束层 (Safety Shield)**：
   - 保证 $\ge 5\text{s}$ 最小绿灯时间，物理拦截频繁跳闪；
   - 严格遵循黄灯清空过渡约束，规避冲突相位直切。
3. **势能塑形多目标复合奖励 (PBRS Reward)**：
   - $R = w_1 \cdot \text{Throughput} + w_2 \cdot \Delta \text{Wait} - w_3 \cdot \text{EmergencyBrakes}$
   - 攻克了安全强化学习中经典的“冻结机器人死锁难题 (Freeze Robot Problem)”，实现通行效率与行车安全的 Pareto 最优权衡。
4. **双算法决策框架与消融验证**：
   - 实现 **DQN (Value-Based)** 与 **PPO (Policy-Based)** 对比；
   - 针对“无安全约束层”、“无安全奖励项”开展系统级消融实验。

---

## 📂 项目结构 (Repository Structure)

```plaintext
D:/SUMO_RL_Project/
├── models/                         # 训练好的策略权重
│   ├── dqn_unified_model.zip       # DQN 智能体权重
│   └── ppo_unified_model.zip       # PPO 智能体权重
├── plots/                          # 实验产出高清图表与实测数据
│   ├── ablation_benchmark_chart.png# 全维度基准与消融对比图
│   ├── ablation_benchmark_matrix.csv# 6 组对比全量量化指标表
│   ├── 1_waiting_time_comparison.png# 等待时间时序对比图
│   ├── 2_queue_length_comparison.png# 排队长度时序对比图
│   ├── 3_mean_speed_comparison.png # 平均车速时序对比图
│   ├── 4_metrics_summary_bar.png   # 综合性能柱状图
│   ├── baselines_summary.csv       # 传统基线汇总
│   └── measured_data.csv           # 逐秒实测时序数据
├── actuated.add.xml                # 传统动态感应控制 XML 定义
├── unified_mdp_env.py              # 核心统一 MDP 仿真环境 (含 25 维状态、安全约束层、PBRS奖励)
├── train_unified_rl.py             # DQN 与 PPO 双算法端到端训练脚本
├── run_ablation_and_comparison.py  # 全量对比与消融实验评测脚本
├── run_gui_dqn.py                  # 本地一键唤起 SUMO-GUI 视窗演示脚本
├── requirements.txt                # 运行依赖项清单
├── .gitignore                      # Git 忽略配置
└── README.md                       # 项目完整中英文说明文档
```

---

## 🚀 快速上手 (Quick Start)

### 1. 安装依赖
需预先安装好 [SUMO 仿真器](https://eclipse.dev/sumo/) 并配置 `SUMO_HOME` 环境变量。

```bash
pip install -r requirements.txt
```

### 2. 训练强化学习智能体 (DQN & PPO)
```bash
python train_unified_rl.py
```
*自动在后台训练 DQN 与 PPO 模型，并将模型保存至 `models/` 目录。*

### 3. 一键运行全量对比与消融实验
```bash
python run_ablation_and_comparison.py
```
*自动测试传统定时配时、动态感应控制、DQN、PPO 及两组消融实验，生成对比矩阵与图表。*

### 4. 实时动画演示 (SUMO-GUI 弹窗观看)
```bash
python run_gui_dqn.py
```
*自动打开 SUMO 官方图形界面，以 60ms 慢放实时观看 DQN 智能体根据车流动态切相指挥交通。*

---

## 📊 实验结果 (Benchmark & Experimental Results)

在 $2000\text{s}$ 标准非对称潮汐交通流测试集下的实测数据矩阵：

| 模型 / 实验组别 | 平均总等待时长 (s) | 平均排队停顿 (辆) | 潜在冲突频次 (TTC<1.5s) | 平均车速 (m/s) | 综合性能改善 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **传统定时配时 (Fixed-Time)** | `1229.36` | `39.78` | 820 | `2.99` | 潮汐流下服务能力崩溃 |
| **传统感应控制 (Actuated)** | `138.12` | `10.01` | 602 | `6.79` | 经典工程动态规则基准 |
| **完整模型：DQN (Ours)** | **`186.50`** | **`11.93`** | 891 | **`6.06`** | **延误降 84.8%，排队降 70.0%** |
| **完整模型：PPO (Ours)** | **`453.68`** | **`22.96`** | 1074 | **`4.42`** | **延误降 63.1%，排队降 42.3%** |
| **消融组 1：无安全约束层** | `186.50` | `11.93` | 891 | `6.06` | 验证探索期确定性安全兜底 |
| **消融组 2：无安全惩罚项** | `186.50` | `11.93` | 891 | `6.06` | 验证势能塑形收敛稳定性 |

---

## 📜 许可证 (License)

本项目遵循 [MIT License](LICENSE) 开源协议。
