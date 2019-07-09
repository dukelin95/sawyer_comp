
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np
from ik_wrapper import IKWrapper
from gym_goal_wrapper import GymGoalEnvWrapper

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import HER
from stable_baselines import DDPG

#from sawyer_primitive_reach import SawyerPrimitiveReach

from test_sawyer import SawyerPrimitiveReach
import argparse

parser = argparse.ArgumentParser(description='Select pkl file to visualize')
parser.add_argument('path', metavar='path', type=str)
args = parser.parse_args()

policy = 'x'
reward_shaping = True
if reward_shaping:
    print("Policy {0} with dense rewards".format(policy))
else:
    print("Policy {0} with sparse rewards".format(policy))
env = GymGoalEnvWrapper(
       IKWrapper(
        SawyerPrimitiveReach(
            prim_axis=policy,
            has_renderer=True,
            has_offscreen_renderer=False,
      	    use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=reward_shaping,
            horizon = 500,
            control_freq=100,  # control should happen fast enough so that simulation looks smooth
        )
       ),reward_shaping=reward_shaping)

path = args.path
model = HER.load(path, env=env)

for u in range(5):
  print("Trial{}".format(u))
  obs = env.reset()
  done = False
  while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.viewer.viewer.add_marker(pos=env.goal, size=np.array((0.02,0.02,0.02)), label='goal', rgba=[1, 0, 0, 0.5])
    if env.reward() > 0.1: print(env.reward())
    env.render()

