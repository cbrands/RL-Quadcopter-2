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

        self.action_low = 450
        self.action_high = 750
        self.state_size = 18
        self.action_size = 4


    def get_reward(self):
        """Uses current position and velocity to get reward."""

        alphaP = 1.
        alphaV = 1.
        
        reward = 0.
        
        # Reward movement in z direction
        reward += alphaP * (self.sim.pose[2])
        
        #Reward velocity in z direction
        #reward += alphaV * (self.sim.v[2])
        
        # Penalise sideways (x,y) position
        reward += 1 * (1 - np.linalg.norm(self.sim.pose[:2] - [0., 0.]))
        # Penalise sideways (x,y) velocity
        #reward += 1 * (1 - np.linalg.norm(self.sim.v[:2] - [0., 0.]))
        # Penalise angular velocity
        #reward += 1 * (1 - np.linalg.norm(self.sim.angular_v - [0., 0., 0.]))
        # Penalise angular position
        reward += 1 * (1 - np.linalg.norm(self.sim.pose[3:] - [0., 0., 0.]))

        return reward

    def step(self, rotor_speeds):
        """Perform next step."""
        reward = 0
        poses = []
        #rotor_speeds = [rotor_speeds[0], rotor_speeds[0], rotor_speeds[0], rotor_speeds[0]]
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
