"""看训练好的策略怎么控制手和门。"""
import argparse
from pathlib import Path

import gymnasium as gym
import imageio
from stable_baselines3 import PPO

from imitate.envs.register import register_envs
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["human", "video"], default="human")
    parser.add_argument("--model", default="imitate/runs/ppo/final.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="imitate/runs/ppo/videos")
    args = parser.parse_args()

    register_envs()
    render_mode = "human" if args.mode == "human" else "rgb_array"
    env = gym.make("MyDoor-v0", render_mode=render_mode)
    model = PPO.load(args.model, device="cpu")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = env.unwrapped.metadata["render_fps"]

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        frames = [env.render()] if args.mode == "video" else []
        hit = False
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            hit = hit or bool(info["success"])
            if args.mode == "video":
                frames.append(env.render())
            done = terminated or truncated

        if args.mode == "video":
            path = out_dir / f"ep{ep}_success{int(hit)}.mp4"
            imageio.mimsave(path, frames, fps=fps)
            print("saved", path)
        else:
            print(f"episode {ep}: success={hit}")

    env.close()


if __name__ == "__main__":
    main()