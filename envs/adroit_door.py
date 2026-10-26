from pathlib import Path
import numpy as np

from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames

# 相机配置
DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5, # distance：相机到看点的距离。
    "azimuth": 90.0, # azimuth：水平旋转角（度）。
}

class MyDoorEnv(MujocoEnv,EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(self, reward_type: str = "dense", **kwargs):
        xml_file_path = (
            Path(__file__).resolve().parents[2]
            / "imitate"
            / "envs"
            / "assets"
            / "adroit_hand"
            / "adroit_door.xml"
        )
        # 27 (手臂3 + 手24)+1 (门闩)+1 (门铰链)+3+3+3 (掌心、把手、相对向量)+1 (开门标志)=39
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(39,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self,
            model_path = str(xml_file_path),
            frame_skip = 5,
            observation_space = observation_space,
            default_camera_config = DEFAULT_CAMERA_CONFIG,
            **kwargs
        )
        # 把 XML 里的名字，一次性翻译成 MuJoCo 内部的整数 id，存成字典备用。
        self._model_names = MujocoModelNames(self.model)
        # action_space是从Env中继承的，原本存放的是电机控制的范围，单位根据关节的类型不同而不同，这套执行器是位置伺服，可能是弧度、米等。这里将其归一化到[-1, 1]范围内，方便训练。
        # 此处将范围限制在-1，1之间，方便训练。因为在强化学习中，通常使用归一化的动作空间可以提高训练的稳定性和收敛速度。
        self.action_space = spaces.Box(
            low = -1.0, high = 1.0, shape = self.action_space.shape, dtype = np.float64
        )

        # 解析奖励函数
        if reward_type.lower() == "dense":
            self.sparse_reward = False
        elif reward_type.lower() == "sparse":
            self.sparse_reward = True
        else:
            raise ValueError(
                f"Unknown reward type, expected `dense` or `sparse` but got {reward_type}"
            )

        n2id = self._model_names.actuator_name2id
        # 先择n2id["A_WRJ1"] : n2id["A_WRJ0"]这些执行器，每个执行器都配置三个增益系数
        # 这个工作其实已经被 MujocoEnv::self.model, self.data = self._initialize_simulation()执行过了，这里写出来的目的是为了显式固定，防止被修改了
        self.model.actuator_gainprm[n2id["A_WRJ1"] : n2id["A_WRJ0"] + 1, :3] = [10, 0, 0]
        self.model.actuator_gainprm[n2id["A_FFJ3"] : n2id["A_THJ0"] + 1, :3] = [1, 0, 0]

        self.model.actuator_biasprm[n2id["A_WRJ1"] : n2id["A_WRJ0"] + 1, :3] = [0, -10, 0]
        self.model.actuator_biasprm[n2id["A_FFJ3"] : n2id["A_THJ0"] + 1, :3] = [0, -1, 0]
         # 后面归一化要使用到的参数
        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )
        # 缓存后面要反复用的 id
        self.door_hinge_addrs = self.model.jnt_dofadr[
            self._model_names.joint_name2id["door_hinge"]
        ]
        self.grasp_site_id = self._model_names.site_name2id["S_grasp"]
        self.handle_site_id = self._model_names.site_name2id["S_handle"]
        self.door_body_id = self._model_names.body_name2id["frame"]

        # 接触刚度：默认 0.02 太软，200N 会压进 2–3cm
        self.model.geom_solref[:, 0] = 0.005

        # 穿透惩罚要用的物体 body id
        self._object_bodies = {
            self._model_names.body_name2id[name]
            for name in ("frame", "door", "latch")
        }

        '''
        qpos and qvel 的结构：
        0–3	ARTz, ARRx, ARRy, ARRz	4
        4–27	WRJ1 ... THJ0（24 个手关节）	24
        28	door_hinge（门铰链角）	1
        29	latch（门闩角）	1
        door_body_pos是门框就一个位置，他动了，它里面的东西也跟着动
        '''
        self._state_space = spaces.Dict(
            {
                "qpos": spaces.Box(-np.inf, np.inf, shape=(30,), dtype=np.float64),
                "qvel": spaces.Box(-np.inf, np.inf, shape=(30,), dtype=np.float64),
                "door_body_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
            }
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)
    
    '''
    先想清楚要哪 7 段
    顺序	段	维度	来源
    1	手臂 + 手关节角	27	qpos[1:-2]
    2	门闩角	1	qpos[-1]
    3	门铰链角	1	qpos[door_hinge_addr]
    4	掌心位置	3	site_xpos[S_grasp]
    5	把手位置	3	site_xpos[S_handle]
    6	掌心 − 把手	3	4 − 5
    7	开门标志	1	门角 > 1.0 ?        
    '''
    def _get_obs(self):
        # 目前我还没有看到什么地方个
        qpos = self.data.qpos.ravel() # numpy的方法，展开成一维数组

        palm_pos = self.data.site_xpos[self.grasp_site_id].ravel()
        handle_pos = self.data.site_xpos[self.handle_site_id].ravel()

        door_angle = qpos[self.door_hinge_addrs]
        latch_angle = qpos[-1]
        door_open = 1.0 if door_angle > 1.0 else -1.0

        return np.concatenate(
                [
                    qpos[1:-2],              # 27
                    [latch_angle],           # 1
                    [door_angle],            # 1
                    palm_pos,                # 3
                    handle_pos,              # 3
                    palm_pos - handle_pos,   # 3
                    [door_open],             # 1
                ]
            )

    def step(self, action):
        # 1. 动作：先裁剪，再映射到物理区间
        action = np.clip(action, -1.0, 1.0)
        ctrl = self.act_mean + action * self.act_rng

        # 2. 推进物理：5 个子步
        self.do_simulation(ctrl, self.frame_skip)

        # 3. 读新状态
        obs = self._get_obs()
        door_angle = self.data.qpos[self.door_hinge_addrs]
        success = door_angle >= 1.35

        # 4. 奖励
        if self.sparse_reward:
            reward = 10.0 if success else -0.1
        else:
            palm_pos = self.data.site_xpos[self.grasp_site_id].ravel()
            handle_pos = self.data.site_xpos[self.handle_site_id].ravel()

            # reward = -0.1 * np.linalg.norm(palm_pos - handle_pos)
            dist = np.linalg.norm(palm_pos - handle_pos)
            reward = -0.1 * min(dist, 0.05)      # 5cm 以内不再加分

            reward += -0.1 * (door_angle - 1.57) ** 2
            reward += -1e-5 * np.sum(self.data.qvel**2)

            if door_angle > 0.2:
                reward += 2
            if door_angle > 1.0:
                reward += 8
            if door_angle > 1.35:
                reward += 10

            penetration = 0.0
            for k in range(self.data.ncon):
                c = self.data.contact[k]
                if c.dist < -0.005:              # 允许 5mm 正常压缩
                    b1 = self.model.geom_bodyid[c.geom1]
                    b2 = self.model.geom_bodyid[c.geom2]
                    # if b1 in self._object_bodies or b2 in self._object_bodies:
                    if (b1 in self._object_bodies) != (b2 in self._object_bodies):
                        penetration += -c.dist - 0.005
            reward -= 200.0 * penetration

        # 5. human 模式才自动渲染
        if self.render_mode == "human":
            self.render()
        return obs, reward, False, False, {"success": success}
    
    # 在MujocoEnv的reset函数里面调用了reset_model
    def reset_model(self):
        self.model.body_pos[self.door_body_id, 0] = self.np_random.uniform(-0.3, -0.2)
        self.model.body_pos[self.door_body_id, 1] = self.np_random.uniform(0.25, 0.35)
        self.model.body_pos[self.door_body_id, 2] = self.np_random.uniform(0.252, 0.35)
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()


    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed)

        if options is not None and "initial_state_dict" in options:
            self.set_env_state(options["initial_state_dict"])
            obs = self._get_obs()

        return obs, info

    def get_env_state(self):
        return dict(
            qpos=self.data.qpos.ravel().copy(),
            qvel=self.data.qvel.ravel().copy(),
            door_body_pos=self.model.body_pos[self.door_body_id].ravel().copy(),
        )

    def set_env_state(self, state_dict):
        assert self._state_space.contains(state_dict), (
            f"The state dictionary {state_dict} must be a member of {self._state_space}."
        )

        self.model.body_pos[self.door_body_id] = state_dict["door_body_pos"]
        self.set_state(state_dict["qpos"], state_dict["qvel"])