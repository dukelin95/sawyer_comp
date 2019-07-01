import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG

env = gym.make("MountainCarContinuous-v0")

param_noise = None
action_noise = None
model = DDPG('MlpPolicy', env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=10000)
#model.save("ddpg_mountain")
#del model # remove to demonstrate saving and loading

#model = DDPG.load("ddpg_mountain")

#for u in range(100):
#  i = 0
#  obs = env.reset()
#  while i != 1000:
#    i = i + 1
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
#    env.render()
#
