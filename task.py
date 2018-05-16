import numpy as np
from physics_sim import PhysicsSim
import math

class Task():
    """Task, define goal and rewards the agent based on result."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, time_limit=10., target_pos=None):
        """Initialize a Task object."""
        
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, time_limit)
        self.action_repeat = 3

        self.action_low = 10
        self.action_high = 800
        self.state_size = 18
        self.action_size = 4

        # Target
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10., 0., 0., 0.])
        self.target_vel = np.array([0., 0., 0., 0., 0., 0.])

    def get_reward(self):
        """Uses current position and velocity to get reward."""

        alphaP = 0.3
        alphaV = 0.03
        alphaA = 0.01
        
        reward = alphaP * (1 - np.tanh(np.linalg.norm(self.sim.pose[:3] - self.target_pos[:3])))
        reward += alphaV * (1 - np.tanh(np.linalg.norm(self.sim.v - self.target_vel[:3])))
        reward += alphaA * (1 - np.tanh(np.linalg.norm(self.sim.angular_v - self.target_vel[3:])))

        return reward

    def step(self, rotor_speeds):
        """Perform next step."""
        reward = 0
        poses = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward()
            poses.append(self.sim.pose)
        next_state = np.concatenate(poses)
        return next_state, reward, done

    def reset(self):
        """Start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        self.num_steps = 0
        return state
