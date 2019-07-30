"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np
from gym import spaces
from robosuite.wrappers import Wrapper


class GymGoalEnvWrapper(Wrapper):
    env = None
 
    def __init__(self, env, keys=None, early=False, train=False):
        """
        Initializes the Gym wrapper.

        Args:
            env (MujocoEnv instance): The environment to wrap.
            keys (list of strings): If provided, each observation will
                consist of concatenated keys from the wrapped environment's
                observation dictionary. Defaults to robot-state and object-state.
        """
        super().__init__(env)

        self.early = early
        self.train = train

        self.metadata = None
        if keys is None:
            assert self.env.use_object_obs, "Object observations need to be enabled."
            keys = ["robot-state", "object-state"]
        self.keys = keys

        # set up observation and action spaces
        self.action_space = self.env.action_space
        flat_ob = self._flatten_obs(self.env.reset(), verbose=True)
        self.obs_dim = flat_ob.size


# TODO delete

        # if train, do a smaller space for some observations
        if self.train:
            # order: gripper_qpos, eef_pos, cube_pos
            xyz_high = [0.5 + self.arm_oprange[1], self.arm_oprange[1], 1.0]
            xyz_low = [0.5 + self.arm_oprange[0], self.arm_oprange[0], 0.8]
            high = [0.0115, 0.02]
            low = [-0.02, -0.0115]
            high = high.extend(xyz_high + xyz_high)
            low = low.extend(xyz_low + xyz_low)
            high = np.array(high)
            low = np.array(low)
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(low=np.array(xyz_low), high=np.array(xyz_high)),
                achieved_goal=spaces.Box(low=np.array(xyz_low), high=np.array(xyz_high)),
                observation=spaces.Box(low=low, high=high),
            ))
        # else test, do a bigger space for some observations
        else:
            xyz_high = [0.5 + self.arm_oprange[1], self.arm_oprange[1], 1.0]
            xyz_low = [0.5 + self.arm_oprange[0], self.arm_oprange[0], 0.8]
            high = [0.0115, 0.02]
            low = [-0.02, -0.0115]
            high = high.extend(xyz_high + xyz_high)
            low = low.extend(xyz_low + xyz_low)
            high = np.array(high)
            low = np.array(low)
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(low=np.array(xyz_low), high=np.array(xyz_high)),
                achieved_goal=spaces.Box(low=np.array(xyz_low), high=np.array(xyz_high)),
                observation=spaces.Box(low=low, high=high),
            ))


    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict: ordered dictionary of observations
        """
        ob_lst = []
        for key in obs_dict:
            if key in self.keys:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(obs_dict[key])
        return np.concatenate(ob_lst)

    def reset(self):
        ob_dict = self.env.reset()
        return self.env.get_goalenv_dict(ob_dict)

    def step(self, action):
        ob_dict, reward, done, info = self.env.step(action)
        if reward == 0 and self.early:
            print("early termination")
            done = True
        return self.env.get_goalenv_dict(ob_dict), reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        return self.env.compute_reward(achieved_goal, desired_goal, None)


