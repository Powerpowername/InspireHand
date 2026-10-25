"""第一次运行 MyDoorEnv：形状、步进、状态存取。"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from imitate.envs.adroit_door import MyDoorEnv


def main() -> None:
    env = MyDoorEnv(reward_type="dense")

    print("obs space  ", env.observation_space)
    print("act space  ", env.action_space)

    obs, info = env.reset(seed=0)
    print("reset obs  ", obs.shape, obs.dtype)
    print("in space   ", env.observation_space.contains(obs))
    print("reset info ", info)

    total = 0.0
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        print(
            f"step {i}: r={reward:+.4f} "
            f"term={terminated} trunc={truncated} success={info['success']}"
        )

    print("5-step return", round(total, 4))

    state = env.get_env_state()
    env.step(env.action_space.sample())
    env.set_env_state(state)
    obs2 = env._get_obs()
    print("state roundtrip", np.allclose(obs2, obs))

    env.close()


if __name__ == "__main__":
    main()