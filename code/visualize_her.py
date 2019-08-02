
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np
from ik_wrapper import IKWrapper
from gym_goal_wrapper import GymGoalEnvWrapper

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import HER
from stable_baselines import DDPG

import argparse

parser = argparse.ArgumentParser(description='Select pkl file to visualize and env')
parser.add_argument('envir', metavar='envir', type=str)
parser.add_argument('path', metavar='path', type=str)
args = parser.parse_args()

early = False

if args.envir == 'xyz':
    from test_sawyer_xyz import SawyerPrimitiveReach
    from param_xyz_env import *
    Environment = SawyerPrimitiveReach(
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
    gripper = False
elif args.envir == 'pick':
    from test_sawyer_pick import SawyerPrimitivePick
    from param_pick_env import *
    instructive = 0.0
    random_arm_init=True
    Environment = SawyerPrimitivePick(
            instructive=instructive,
            random_arm_init=random_arm_init,
            has_renderer=True,
            has_offscreen_renderer=False,
      	    use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=reward_shaping,
            horizon = 100,
            control_freq=100,  # control should happen fast enough so that simulation looks smooth
        )
    gripper = True
else:
    raise Exception(
        "Only xyz or pick accepted"
    )


#if reward_shaping:
#    print("Policy {0} with dense rewards".format(policy))
#else:
#    print("Policy {0} with sparse rewards".format(policy))
env = GymGoalEnvWrapper(
       IKWrapper(
        Environment, gripper=gripper
       ), early=early)

path = args.path
model = HER.load(path, env=env)

succ = 0
loop = 10
for u in range(loop):
  print("Trial{}".format(u))
  obs = env.reset()
  print(env.goal)
  done = False
  while not done:
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, done, info = env.step(action)
    env.viewer.viewer.add_marker(pos=env.goal, size=np.array((0.02,0.02,0.02)), label='goal', rgba=[1, 0, 0, 0.5])
    if env.reward() == 0.0: 
       succ = succ + 1
       print('Success: {}'.format(obs['achieved_goal']))
#       break
    env.render()
print('{0}/{1}'.format(succ, loop))
