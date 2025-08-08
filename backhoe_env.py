import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

class BackhoeHydraulicEnv(gym.Env):
    """Simplified backhoe arm with hydraulic-like dynamics.

    This environment models a turret, boom, stick and bucket with a bucket
    cylinder that is mimicked to the bucket joint.  A PD controller with
    spring-damper terms approximates hydraulic behavior.  It is *not* a full
    fluid simulation but captures basic inertial and coupling effects.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render: bool = False):
        super().__init__()
        self.render_mode = render
        self.time_step = 1.0 / 240.0

        # turret, boom, stick, bucket cylinder extension
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # joint positions (4) + joint velocities (4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        self.physics_client = None
        self.arm_joint_indices = [0, 1, 2, 3]
        self.arm_init_positions = [0.0, -0.3, 0.8, 0.2]

        # Dynamics tuning
        self.joint_damping = 2.0
        self.joint_friction = 0.5
        self.joint_limits = None

        # PD gains
        self.Kp = np.array([220, 250, 230, 180])
        self.Kd = np.array([35, 40, 35, 20])
        # Spring-damper constants for hydraulic approximation
        self.spring_k = np.array([60, 80, 70, 50])
        self.damp_b = np.array([6, 8, 7, 5])

    def _load_environment(self) -> None:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)

        urdf_path = os.path.join(os.path.dirname(__file__), "real_backhoe.urdf")
        self.arm_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        limits = []
        for idx, pos in zip(self.arm_joint_indices, self.arm_init_positions):
            p.resetJointState(self.arm_id, idx, targetValue=pos, targetVelocity=0.0)
            info = p.getJointInfo(self.arm_id, idx)
            limits.append((info[8], info[9]))
            p.changeDynamics(
                self.arm_id,
                idx,
                jointDamping=self.joint_damping,
                jointFriction=self.joint_friction,
            )
        self.joint_limits = np.array(limits)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.physics_client is None:
            self.physics_client = p.connect(p.GUI if self.render_mode else p.DIRECT)
            p.setTimeStep(self.time_step)

        p.resetSimulation()
        self._load_environment()

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        joint_states = p.getJointStates(self.arm_id, self.arm_joint_indices)
        angles = np.array([s[0] for s in joint_states])
        velocities = np.array([s[1] for s in joint_states])

        desired = np.clip(
            angles + action * 0.05, self.joint_limits[:, 0], self.joint_limits[:, 1]
        )
        error = desired - angles

        control = self.Kp * error - self.Kd * velocities
        spring = -self.spring_k * error
        damper = -self.damp_b * velocities
        torque = control + spring + damper
        torque = np.clip(torque, -800, 800)

        for i, (idx, tau) in enumerate(zip(self.arm_joint_indices, torque)):
            lower, upper = self.joint_limits[i]
            angle = angles[i]
            if angle <= lower + 0.01 and tau < 0:
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm_id,
                    jointIndex=idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=lower,
                    force=2000,
                )
            elif angle >= upper - 0.01 and tau > 0:
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm_id,
                    jointIndex=idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=upper,
                    force=2000,
                )
            else:
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm_id,
                    jointIndex=idx,
                    controlMode=p.TORQUE_CONTROL,
                    force=float(tau),
                )

        p.stepSimulation()
        obs = self._get_observation()
        reward = -0.05 * np.sum(error ** 2) - 0.01 * np.sum(velocities ** 2)
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        joint_states = p.getJointStates(self.arm_id, self.arm_joint_indices)
        positions = [s[0] for s in joint_states]
        velocities = [s[1] for s in joint_states]
        return np.array(positions + velocities, dtype=np.float32)

    def render(self):
        # GUI handled by pybullet automatically
        pass

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
