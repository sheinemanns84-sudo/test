import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os


class BackhoeHydraulicEnv(gym.Env):
    """Baggerarm-Umgebung mit hydraulikähnlicher Steuerung und Zielhaltung."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render: bool = False):
        super().__init__()
        self.render_mode = render
        self.time_step = 1.0 / 240.0

        self.physics_client = None
        self.arm_id = None
        self.arm_joint_indices = []
        self.arm_init_positions = []

        # Platzhalter bis die Gelenke geladen sind
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Reglerparameter für bis zu vier Gelenke (Turret, Boom, Stick, Bucket)
        self.Kp_full = np.array([220, 250, 230, 180])
        self.Kd_full = np.array([35, 40, 35, 20])
        self.spring_k_full = np.array([60, 80, 70, 50])
        self.damp_b_full = np.array([6, 8, 7, 5])

        self.hold_position = None  # für PD-Halteverhalten
        self.goal_position = None  # Zielpose für die Belohnungsfunktion

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

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self.physics_client is None:
            self.physics_client = p.connect(p.GUI if self.render_mode else p.DIRECT)
            p.setTimeStep(self.time_step)

        p.resetSimulation()
        p.setTimeStep(self.time_step)  # ensure custom step persists after reset
        self._load_environment()

        self.arm_joint_indices = [
            i for i in range(p.getNumJoints(self.arm_id))
            if p.getJointInfo(self.arm_id, i)[2] != p.JOINT_FIXED
        ]

        num_joints = len(self.arm_joint_indices)

        # Realistische Parkposition
        self.arm_init_positions = [
            0.0,   # Turret – geradeaus
            0.8,   # Boom – leicht angehoben
            -1.0,  # Stick – zurückgezogen
            -0.5   # Bucket – halb geschlossen
        ][:num_joints]

        for idx, pos in zip(self.arm_joint_indices, self.arm_init_positions):
            p.resetJointState(self.arm_id, idx, targetValue=pos, targetVelocity=0.0)

        # Observation/Action-Spaces aktualisieren
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * num_joints,), dtype=np.float32)

        self.Kp = self.Kp_full[:num_joints]
        self.Kd = self.Kd_full[:num_joints]
        self.spring_k = self.spring_k_full[:num_joints]
        self.damp_b = self.damp_b_full[:num_joints]

        self.hold_position = np.array(self.arm_init_positions, dtype=np.float32)
        self.goal_position = np.array(self.arm_init_positions, dtype=np.float32)

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        joint_states = p.getJointStates(self.arm_id, self.arm_joint_indices)
        angles = np.array([s[0] for s in joint_states])
        velocities = np.array([s[1] for s in joint_states])

        deadzone = 0.05
        is_active = np.abs(action) > deadzone

        if np.any(is_active):
            self.hold_position = angles.copy()
            Pa_in_Bar = 10 ** 7
            MaxBar = 8 * Pa_in_Bar
            Zylinderflaeche = np.array([0.2, 0.196, 0.146, 0.140])[:len(action)]
            Hebelarm = np.array([2.0, 3.6, 1.6, 0.5])[:len(action)]

            Druck = action * MaxBar
            Kraft = Druck * Zylinderflaeche
            torque = Kraft * Hebelarm
        else:
            error = self.hold_position - angles
            torque = self.Kp * error - self.Kd * velocities

        damper = -self.damp_b * velocities
        torque += damper
        torque = np.clip(torque, -80000, 80000)

        for idx, tau in zip(self.arm_joint_indices, torque):
            p.setJointMotorControl2(
                bodyUniqueId=self.arm_id,
                jointIndex=idx,
                controlMode=p.TORQUE_CONTROL,
                force=float(tau),
            )

        p.stepSimulation()

        obs = self._get_observation()
        goal_error = angles - self.goal_position
        reward = -0.05 * np.sum(goal_error ** 2) - 0.01 * np.sum(velocities ** 2)
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        joint_states = p.getJointStates(self.arm_id, self.arm_joint_indices)
        positions = [s[0] for s in joint_states]
        velocities = [s[1] for s in joint_states]
        return np.array(positions + velocities, dtype=np.float32)

    def render(self):
        pass  # PyBullet GUI läuft automatisch

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
