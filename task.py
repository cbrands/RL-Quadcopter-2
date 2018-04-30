import numpy as np
from physics_sim import PhysicsSim
import math

class Task():
    """
    Task (environment) that defines the goal and provides feedback to the agent.

    Task definition
    I want the quadcopter to take off and hover at position x = 3, y = 5, z = 20.
    I want the orientation of the quadcopter to still be the same as it was at the start.
    I want it to be stable meaning that at the end of the time limit the velocity and angular velocity should be near 0.
    The velocity is less important than the position. With other words as long as the quadcopter is not at the
    the target position it should move.
    """
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None, target_vel=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        #self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.target_pos = target_pos if target_pos is not None else np.array([3., 5., 20., 0., 0., 0.])
        self.target_vel = target_vel if target_vel is not None else np.array([0., 0., 0., 0., 0., 0.])



    def get_reward(self):
        """Uses current pose of sim to return reward."""
        alpha = 0.1
        beta = 0.01
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        pos_reward = sigmoid(np.linalg.norm(self.sim.pose[:3] - self.target_pos[:3]))
        vel_reward = sigmoid(np.linalg.norm(self.sim.v - self.target_vel[:3]))
        ang_reward = sigmoid(np.linalg.norm(self.sim.angular_v - self.target_vel[3:]))

        reward_pos = alpha * (0.5 - sigmoid(pos_reward))
        reward_vel = beta * (0.5 - sigmoid(vel_reward)) + beta * (0.5 - sigmoid(ang_reward))
        return reward_pos + reward_vel

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
