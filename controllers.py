"""Construct a fresh decision function for a benchmark or playback episode."""

from pathlib import Path


CHECKPOINTS = {"dqn": "dqn_unified_model.zip", "ppo": "ppo_unified_model.zip",
               "adaptive-dqn": "adaptive_dqn.zip"}
PLAYBACK_CONTROLLERS = (*CHECKPOINTS, "queue", "predictive", "learned-phase")


def load_checkpoint(name):
    from stable_baselines3 import DQN, PPO

    path = Path(__file__).resolve().parent / "models" / CHECKPOINTS[name]
    return (PPO if name == "ppo" else DQN).load(path, device="cpu")


def episode_policy(name, model=None):
    if name in ("fixed", "actuated"):
        return None  # SUMO advances the native signal program.
    if name in CHECKPOINTS:
        if model is None:
            model = load_checkpoint(name)
        return lambda env, obs: int(model.predict(obs, deterministic=True)[0].item())
    if name == "queue":
        from queue_controller import QueueController
        policy = QueueController()
    elif name == "predictive":
        from predictive_controller import PredictiveController
        policy = PredictiveController()
    elif name == "learned-phase":
        from phase_policy import LearnedPhaseController
        policy = LearnedPhaseController()
    else:
        raise ValueError(f"Unknown controller: {name}")
    return lambda env, obs: policy.predict(env)
