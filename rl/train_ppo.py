"""用 PPO 训练 MyDoor-v0。"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym

from envs.register import register_envs
EPOCHS = 500
# 每个 epoch 训练多少步，PPO 会把这些步分成 mini-batch 训练。这里设置为 16 个 env，每个 env 2048 步，总共 32768 步。
STEPS_PER_EPOCH = 2048 * 16      # 32768

def make_env(rank: int, seed: int = 0,pro: str = ""):
    def _init():
        register_envs(pro)                 # 子进程里也要注册
        if pro == "adroit":
            env = gym.make("AdroitHammer-v0")     # 自带 TimeLimit(200)
        elif pro == "mydoor":
            env = gym.make("MyDoor-v0")     # 自带 TimeLimit(200)
        env.reset(seed=seed + rank)
        return env

    return _init


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    parser = argparse.ArgumentParser()  
    parser.add_argument("--timesteps", type=int, default=EPOCHS * STEPS_PER_EPOCH)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pro", type=str, default="adroit")
    args = parser.parse_args()
    save_dir = Path("/runs/ppo")
    save_dir.mkdir(parents=True, exist_ok=True)
    # SubprocVecEnv创建训练子进程需要的函数
    venv = SubprocVecEnv([make_env(i, args.seed, args.pro) for i in range(args.n_envs)])
    # VecMonitor包装器可以记录每个子进程的 episode reward 和 episode length
    venv = VecMonitor(venv)

    model = PPO(
        "MlpPolicy",
        venv,
        device=args.device,
        n_steps=2048,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=1e-3,
        learning_rate=3e-4,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256], log_std_init=-0.5),
        tensorboard_log=str(save_dir / "tb"),
        verbose=1,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(str(save_dir / "final"))
    venv.close()