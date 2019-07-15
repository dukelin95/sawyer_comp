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
 
    def __init__(self, env, keys=None, reward_shaping=True):
        """
        Initializes the Gym wrapper.

        Args:
            env (MujocoEnv instance): The environment to wrap.
            keys (list of strings): If provided, each observation will
                consist of concatenated keys from the wrapped environment's
                observation dictionary. Defaults to robot-state and object-state.
        """
        self.env = env
        self.metadata = None
        if keys is None:
            assert self.env.use_object_obs, "Object observations need to be enabled."
            keys = ["robot-state", "object-state"]
        self.keys = keys

        # set up observation and action spaces
        flat_ob = self._flatten_obs(self.env.reset(), verbose=True)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Dict(dict(
           desired_goal=spaces.Box(-np.inf, np.inf, shape=(3,)),
           achieved_goal=spaces.Box(-np.inf, np.inf, shape = (3,)),
           observation=spaces.Box(low=low, high=high,),
        ))

        low = -np.array([.01, .01, .01])
        high = np.array([.01, .01, .01])

        self.action_space = spaces.Box(low=low, high=high)

        self.reward_shaping = reward_shaping

    def _get_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information. Return as goal env dict

        Args:
            obs_dict: ordered dictionary of observations
        """
        ob_lst = []
        di = {}
        for key in obs_dict:
            if key in self.keys:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(obs_dict[key])
        di['observation'] = np.concatenate(ob_lst)
        di['desired_goal'] = obs_dict['object-state'][0:3]
        di['achieved_goal'] = obs_dict['robot-state'][23:26]

        return di

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
        return self._get_obs(ob_dict)

    def step(self, action):
        ob_dict, reward, done, info = self.env.step(action)
        return self._get_obs(ob_dict), reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        velocity_pen = 0.0
        distance_threshold = 0.05
        d = np.linalg.norm(achieved_goal - desired_goal)
        if self.reward_shaping: # dense
            reward = 1 - np.tanh(10 * d)
            if d <= distance_threshold:
                reward = 10.0
        else: # sparse (-1 or 0)
            #reward = -np.float32(d > distance_threshold)
            if d > distance_threshold:
                reward = -1.0
            else:
                reward = 0.0 
       
        return reward - velocity_pen


