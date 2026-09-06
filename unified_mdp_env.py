import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sumo_rl
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal

class UnifiedIntersectionObservation(ObservationFunction):
    """
    状态空间定制类：
    包含四大核心状态特征：
    1. 各车道排队长度 (Queue Length, 8维)
    2. 各车道平均车头时距 (Headway, 8维)
    3. 相位剩余时间与当前相位 One-hot (Remaining Phase Time, 1 + 4 = 5维)
    4. 进口道潜在冲突风险指数 (Conflict Risk / TTC, 4维)
    总状态维度: 8 + 8 + 5 + 4 = 25 维
    """
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)
        self.max_headway = 50.0  # 最大参考车头间距 (米)
        self.ttc_threshold = 2.0 # 冲突临界 TTC 阈值 (秒)

    def __call__(self) -> np.ndarray:
        # 1. 各车道排队长度 (归一化到 [0, 1])
        queues = self.ts.get_lanes_queue()
        
        # 2. 各车道平均车头间距 (Headway)
        headways = []
        for lane in self.ts.lanes:
            vehs = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            if len(vehs) >= 2:
                dists = []
                for i in range(len(vehs) - 1):
                    # 获取前后车相对距离
                    leader = self.ts.sumo.vehicle.getLeader(vehs[i])
                    if leader and leader[0] == vehs[i+1]:
                        dists.append(min(leader[1], self.max_headway))
                headways.append(float(np.mean(dists) / self.max_headway) if dists else 1.0)
            else:
                headways.append(1.0)  # 无车或单车时无跟驰车头时距压迫

        # 3. 当前相位 One-hot 编码与剩余绿灯时间比例
        phase_onehot = [1.0 if self.ts.green_phase == i else 0.0 for i in range(self.ts.num_green_phases)]
        # 剩余有效可用绿灯时间比例
        remaining_ratio = max(0.0, (self.ts.max_green - self.ts.time_since_last_phase_change)) / float(self.ts.max_green)

        # 4. 四个流向的冲突风险指数 (基于反向 TTC: 1 / TTC)
        conflict_risks = [0.0, 0.0, 0.0, 0.0]  # 北、东、南、西四个进口方向
        approaches = {'n_t': 0, 'e_t': 1, 's_t': 2, 'w_t': 3}
        for lane in self.ts.lanes:
            app_key = lane[:3]
            app_idx = approaches.get(app_key, 0)
            vehs = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            for v in vehs:
                leader = self.ts.sumo.vehicle.getLeader(v)
                if leader:
                    l_id, dist = leader
                    v_spd = self.ts.sumo.vehicle.getSpeed(v)
                    l_spd = self.ts.sumo.vehicle.getSpeed(l_id)
                    delta_v = v_spd - l_spd
                    if delta_v > 0.1 and dist > 0:
                        ttc = dist / delta_v
                        if ttc < self.ttc_threshold:
                            conflict_risks[app_idx] += min(1.0, 1.0 / (ttc + 0.1))

        # 归一化冲突风险
        conflict_risks = [min(1.0, r / 3.0) for r in conflict_risks]

        # 拼接全状态向量 (8 + 8 + 4 + 1 + 4 = 25 维)
        state_vec = np.array(
            queues + headways + phase_onehot + [remaining_ratio] + conflict_risks,
            dtype=np.float32
        )
        return state_vec

    def observation_space(self) -> spaces.Box:
        dim = 8 + 8 + self.ts.num_green_phases + 1 + 4
        return spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)


class UnifiedMDPIntersectionEnv(gym.Env):
    """
    统一 MDP 交叉口仿真环境：
    1. 整合车道排队、车头时距、相位倒计时、TTC 冲突风险状态
    2. 内置动作掩码 (Action Mask) 与安全约束层 (Safety Shield)
    3. 多权重奖励函数：w1*通行效率 - w2*等待时间 - w3*安全冲突次数
    """
    def __init__(
        self,
        use_gui=False,
        num_seconds=20000,
        delta_time=5,
        min_green=5,
        max_green=50,
        w_efficiency=1.0,   # 通行效率权重
        w_delay=0.1,        # 延误等待惩罚权重
        w_conflict=5.0,     # 安全冲突惩罚权重 (强安全约束)
        enable_safety_shield=True, # 是否开启安全约束层
        sumo_seed=42,
        additional_sumo_cmd=None,
    ):
        super().__init__()
        net_dir = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "2way-single-intersection")
        net_file = os.path.join(net_dir, "single-intersection.net.xml")
        route_file = os.path.join(net_dir, "single-intersection-vhvh.rou.xml")

        self.w_efficiency = w_efficiency
        self.w_delay = w_delay
        self.w_conflict = w_conflict
        self.enable_safety_shield = enable_safety_shield
        self.min_green = min_green

        # 创建基础 sumo-rl 环境，注入自定义 ObservationFunction
        self.env = sumo_rl.SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=use_gui,
            num_seconds=num_seconds,
            single_agent=True,
            delta_time=delta_time,
            min_green=min_green,
            max_green=max_green,
            observation_class=UnifiedIntersectionObservation,
            sumo_seed=sumo_seed,
            additional_sumo_cmd=additional_sumo_cmd,
        )

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.ts_id = self.env.unwrapped.ts_ids[0]

    def get_action_mask(self) -> np.ndarray:
        """
        动作掩码 (Action Mask):
        如果当前绿灯时长还未达到 min_green，只允许保持当前相位 (mask 为 True)，
        禁止切换到其他任何相位 (mask 为 False)，从底层杜绝频繁绿灯闪跳与短绿灯危险。
        """
        ts = self.env.unwrapped.traffic_signals[self.ts_id]
        mask = np.ones(self.action_space.n, dtype=bool)
        if ts.time_since_last_phase_change < self.min_green:
            mask[:] = False
            mask[ts.green_phase] = True  # 仅允许保持当前相位
        return mask

    def reset(self, seed=None, options=None):
        self.last_wait = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        ts = self.env.unwrapped.traffic_signals[self.ts_id]

        # ---------------- 安全约束层 (Safety Constraint Layer / Shield) ----------------
        if self.enable_safety_shield:
            mask = self.get_action_mask()
            # 若智能体输出了不合法的危险动作（被掩码拦截），安全层强制纠偏为安全动作
            if not mask[action]:
                action = ts.green_phase  # 强制保持当前绿灯，确保最小绿灯安全清空

        # 记录执行前的路网累计流出车辆，用于核算这 5 秒内的通过量 (效率)
        prev_departed = ts.sumo.simulation.getDepartedNumber()
        prev_arrived = ts.sumo.simulation.getArrivedNumber()

        # 执行仿真步
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # ---------------- 统计多目标指标 ----------------
        # 1. 通行效率 (本步通过车辆数)
        cur_arrived = ts.sumo.simulation.getArrivedNumber()
        step_throughput = max(0, cur_arrived - prev_arrived)

        # 2. 等待时长
        waiting_time = info.get("system_total_waiting_time", 0)

        # 3. 安全冲突次数 (统计进口道内 TTC < 1.5s 的危险对数)
        conflict_count = 0
        for lane in ts.lanes:
            vehs = ts.sumo.lane.getLastStepVehicleIDs(lane)
            for v in vehs:
                leader = ts.sumo.vehicle.getLeader(v)
                if leader:
                    l_id, dist = leader
                    v_spd = ts.sumo.vehicle.getSpeed(v)
                    l_spd = ts.sumo.vehicle.getSpeed(l_id)
                    delta_v = v_spd - l_spd
                    if delta_v > 0.5 and dist > 0:
                        ttc = dist / delta_v
                        if ttc < 1.5:  # 交通安全常用近邻冲突阈值 1.5 秒
                            conflict_count += 1

        # ---------------- 多权重复合奖励函数计算 ----------------
        # 采用势能塑形 (PBRS)：等待时间减少为正奖励，增加为负惩罚
        cur_wait = float(sum(ts.get_accumulated_waiting_time_per_lane())) / 100.0
        diff_waiting = getattr(self, "last_wait", cur_wait) - cur_wait
        self.last_wait = cur_wait

        # 紧急减速 (Decel < -4.5 m/s^2) 险情统计
        hard_brakes = 0
        for lane in ts.lanes:
            vehs = ts.sumo.lane.getLastStepVehicleIDs(lane)
            for v in vehs:
                try:
                    if ts.sumo.vehicle.getAcceleration(v) < -4.5:
                        hard_brakes += 1
                except Exception:
                    pass

        # R = w1 * 通行效率 + w2 * 延误消解量 - w3 * 危险急刹/冲突惩罚
        reward = (
            self.w_efficiency * float(step_throughput)
            + self.w_delay * float(diff_waiting)
            - self.w_conflict * float(hard_brakes)
        )

        # 把指标注入 info
        info["step_throughput"] = step_throughput
        info["system_waiting_time"] = waiting_time
        info["conflict_count"] = conflict_count
        info["hard_brakes"] = hard_brakes
        info["action_mask"] = self.get_action_mask()

        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()
