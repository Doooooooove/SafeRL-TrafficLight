import os
import sumo_rl
from stable_baselines3 import DQN

def run_gui_dqn(num_seconds=1000):
    """
    弹窗打开 SUMO-GUI 观看训练好的 DQN 智能体实时指挥交通
    """
    net_dir = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "2way-single-intersection")
    net_file = os.path.join(net_dir, "single-intersection.net.xml")
    route_file = os.path.join(net_dir, "single-intersection-vhvh.rou.xml")
    model_path = "D:/SUMO_RL_Project/dqn_sumo_model.zip"

    print("[*] 正在加载训练好的 DQN 模型...")
    model = DQN.load(model_path)

    print("[*] 正在拉起 SUMO-GUI 视窗...")
    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=True,                   # 弹出图形界面
        num_seconds=num_seconds,
        single_agent=True,
        fixed_ts=False,                 # 智能体自主控制
        delta_time=5,
        min_green=5,
        sumo_seed=42,
        additional_sumo_cmd="--delay 60", # 60ms 慢放，方便肉眼观察
    )

    obs, info = env.reset(seed=42)
    done = False

    print("[*] 仿真正在播放中，注意观察智能体如何根据排队灵活切灯...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()
    print("[*] 演示播放结束。")

if __name__ == "__main__":
    run_gui_dqn()
