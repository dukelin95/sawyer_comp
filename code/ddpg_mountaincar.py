import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG

from sawyer_primitive_reach import SawyerPrimitiveReach

# use robosuite's gym_wrapper to wrap the sawyer stack env for baselines

# no object observation for PX,PY,PZ
# TODO turn on for pick policy?
env = gym.make("MountainCar")

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
#action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
action_noise = None
#model = DDPG('MlpPolicy', env, verbose=1, param_noise=param_noise, action_noise=action_noise)

model = DDPG('MlpPolicy', env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=1000)
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
