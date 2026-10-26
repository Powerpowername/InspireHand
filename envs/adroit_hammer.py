from os import path 
from typing import Optional
import numpy as np

from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
from gymnasium_robotics.utils.rotations import quat2euler


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
    "azimuth": 45.0,
}


class AdroitHammerEnv(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, reward_type: str = "dense", **kwargs):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            "../assets/adroit_hand/adroit_hammer.xml",
        )
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(46,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            model_path=xml_file_path,
            frame_skip=5, # 5次mujoco物理引擎计算，调用一次env.step返回/采集一次数据（观测量）
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.model)
        # whether to have sparse rewards
        if reward_type.lower() == "dense":
            self.sparse_reward = False
        elif reward_type.lower() == "sparse":
            self.sparse_reward = True
        else:
            raise ValueError(
                f"Unknown reward type, expected `dense` or `sparse` but got {reward_type}"
            )
        
        # Override action_space to -1, 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape
        )

        # change actuator sensitivity
        self.model.actuator_gainprm[self._model_names.actuator_name2id["A_WRJ1"]: self._model_names.actuator_name2id["A_WRJ0"] + 1,:3,] = np.array([10, 0, 0])
        self.model.actuator_gainprm[self._model_names.actuator_name2id["A_FFJ3"] : self._model_names.actuator_name2id["A_THJ0"] + 1,:3,] = np.array([1, 0, 0])
        self.model.actuator_biasprm[self._model_names.actuator_name2id["A_WRJ1"] : self._model_names.actuator_name2id["A_WRJ0"] + 1,:3,] = np.array([0, -10, 0])
        self.model.actuator_biasprm[self._model_names.actuator_name2id["A_FFJ3"] : self._model_names.actuator_name2id["A_THJ0"] + 1,:3,] = np.array([0, -1, 0])

        self.target_obj_site_id = self._model_names.site_name2id["S_target"]
        self.S_grasp_site_id = self._model_names.site_name2id["S_grasp"]
        self.obj_body_id = self._model_names.body_name2id["Object"]
        self.tool_site_id = self._model_names.site_name2id["tool"]
        self.goal_site_id = self._model_names.site_name2id["nail_goal"]
        self.target_body_id = self._model_names.body_name2id["nail_board"]
        # 归一化系数
        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
        '''        
        0	ARRx	hinge	forearm	大臂上下摆
        1	ARRy	hinge	forearm	大臂左右摆
        2	WRJ1	hinge	wrist	手腕径向/尺偏
        3	WRJ0	hinge	palm	手腕屈/伸
        4–7	FFJ3, FFJ2, FFJ1, FFJ0	hinge	食指	掌指外展、掌指屈、PIP、DIP
        8–11	MFJ3, MFJ2, MFJ1, MFJ0	hinge	中指	同上
        12–15	RFJ3, RFJ2, RFJ1, RFJ0	hinge	无名指	同上
        16–20	LFJ4, LFJ3, LFJ2, LFJ1, LFJ0	hinge	小指	腕掌(CMC)、掌指外展、掌指屈、PIP、DIP
        21–25	THJ4, THJ3, THJ2, THJ1, THJ0	hinge	拇指	CMC 外展、CMC 屈、MCP 外展、MCP 屈、IP
        26	nail_dir	slide	nail	钉子沿板面法向的插入位移（axis="0 0 -1", range="-0.01 0.09"）
        27–29	OBJTx, OBJTy, OBJTz	slide	Object（锤子）	锤子质心的世界 x/y/z 位置
        30–32	OBJRx, OBJRy, OBJRz	hinge	Object（锤子）	锤子绕世界 x/y/z 的旋转角
        '''
        self._state_space = spaces.Dict(
            {
                "qpos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(33,), dtype=np.float64
                ),
                "qvel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(33,), dtype=np.float64
                ),
                "board_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
            }
        )

        EzPickle.__init__(self, **kwargs)
    # a:action 是神经网络训练出来的策略
    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        a = self.act_mean + a * self.act_rng  # mean center and scale

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        hamm_pos = self.data.xpos[self.obj_body_id].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_site_id].ravel()
        head_pos = self.data.site_xpos[self.tool_site_id].ravel()
        nail_pos = self.data.site_xpos[self.target_obj_site_id].ravel()
        goal_pos = self.data.site_xpos[self.goal_site_id].ravel()

        # compute the sparse reward variant first
        goal_distance = np.linalg.norm(nail_pos - goal_pos)
        goal_achieved = goal_distance < 0.01
        reward = 10.0 if goal_achieved else -0.1

        # override reward if not sparse reward
        if not self.sparse_reward:
            # get the palm to the hammer handle
            reward = -0.1 * np.linalg.norm(palm_pos - hamm_pos)
            # take hammer head to nail
            reward -= np.linalg.norm(head_pos - nail_pos)
            # make nail go inside
            reward -= 10 * np.linalg.norm(nail_pos - goal_pos)
            # velocity penalty
            reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())

            # bonus for lifting up the hammer
            if hamm_pos[2] > 0.04 and head_pos[2] > 0.04:
                reward += 2

            # bonus for hammering the nail
            if goal_distance < 0.020:
                reward += 25
            if goal_distance < 0.010:
                reward += 75

        if self.render_mode == "human":
            self.render()

        return obs, reward, False, False, dict(success=goal_achieved)
    
    def _get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        qv = np.clip(self.data.qvel.ravel(), -1.0, 1.0)
        obj_pos = self.data.xpos[self.obj_body_id].ravel()
        obj_rot = quat2euler(self.data.xquat[self.obj_body_id].ravel()).ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_site_id].ravel()
        target_pos = self.data.site_xpos[self.target_obj_site_id].ravel()
        nail_impact = np.clip(
            self.data.sensordata[self._model_names.sensor_name2id["S_nail"]], -1.0, 1.0
        )
        return np.concatenate(
            [
                qp[:-6],
                qv[-6:],
                palm_pos,
                obj_pos,
                obj_rot,
                target_pos,
                np.array([nail_impact]),
            ]
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        obs, info = super().reset(seed=seed)
        if options is not None and "initial_state_dict" in options:
            self.set_env_state(options["initial_state_dict"])
            obs = self._get_obs()

        return obs, info

    def reset_model(self):

        self.model.body_pos[self.target_body_id, 2] = self.np_random.uniform(
            low=0.1, high=0.25
        )
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()
    
    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        board_pos = self.model.body_pos[self.target_body_id].copy()
        target_pos = self.data.site_xpos[self.target_obj_site_id].ravel().copy()
        return dict(qpos=qpos, qvel=qvel, board_pos=board_pos, target_pos=target_pos)

    def set_env_state(self, state_dict: dict[str, np.ndarray]):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        assert all(
            key in state_dict and space.contains(state_dict[key])
            for key, space in self._state_space.spaces.items()
        ), f"The state dictionary {state_dict} must be a member of {self._state_space}."
        qp = state_dict["qpos"]
        qv = state_dict["qvel"]
        board_pos = state_dict["board_pos"]
        self.model.body_pos[self.target_body_id] = board_pos
        self.set_state(qp, qv)