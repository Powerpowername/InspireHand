"""把自写环境注册进 gym，方便 gym.make 使用。"""
import gymnasium as gym
from gymnasium.envs.registration import register

def register_envs(pro : str) -> None:
    if pro == "adroit":
        if "AdroitHammer-v0" in gym.registry:
            return
        
        register(
            id="AdroitHammer-v0",
            entry_point="envs.adroit_hammer:AdroitHammerEnv",
            max_episode_steps=200,
            kwargs={"reward_type": "dense"},
        )
    if pro == "mydoor":
        if "MyDoor-v0" in gym.registry:
            return

        register(
            id="MyDoor-v0",
            entry_point="envs.adroit_door:MyDoorEnv",
            max_episode_steps=200,
            kwargs={"reward_type": "dense"},
    )
    