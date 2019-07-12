
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
reward_shaping = False
limits = [-.15, -.25]
table_full_size = (0.8, 0.8, 0.8)
random_arm_init = False

if reward_shaping:
    print("Policy {0} with dense rewards".format(policy))
else:
    print("Policy {0} with sparse rewards".format(policy))
env = GymGoalEnvWrapper(
       IKWrapper(
        SawyerPrimitiveReach(
            prim_axis=policy,
            limits=limits,
            table_full_size=table_full_size,
            random_arm_init=random_arm_init,
            has_renderer=True,
            has_offscreen_renderer=False,
      	    use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=reward_shaping,
            horizon = 100,
            control_freq=100,  # control should happen fast enough so that simulation looks smooth
        )
       ),reward_shaping=reward_shaping)

path = args.path
model = HER.load(path, env=env)

succ = 0
loop = 100
for u in range(loop):
  print("Trial{}".format(u))
  obs = env.reset()
  print(env.goal)
  done = False
  while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.viewer.viewer.add_marker(pos=env.goal, size=np.array((0.02,0.02,0.02)), label='goal', rgba=[1, 0, 0, 0.5])
    if env.reward() == 0.0: 
       succ = succ + 1
       print(obs['achieved_goal'])
    env.render()
print('{0}/{1}'.format(succ, loop))
