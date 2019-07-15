import time
import robosuite as suite
from gym_goal_wrapper import GymGoalEnvWrapper
from ik_wrapper import IKWrapper

import numpy as np

from stable_baselines import DDPG
from stable_baselines import HER

from test_sawyer_pick import SawyerPrimitivePick
from param_her import *
from param_pick_env import *

import argparse

parser = argparse.ArgumentParser(description='To log or not to log, no for no log')
parser.add_argument('log', metavar='log', type=str)
args = parser.parse_args()
if args.log == 'no':
  print("No saves")
  log = False
else:
  print("All the saves")
  log = True

env = GymGoalEnvWrapper(
       IKWrapper(
        SawyerPrimitivePick(
            random_arm_init=random_arm_init,
            has_renderer=render,
            has_offscreen_renderer=False,
      	    use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=reward_shaping,
            horizon = horizon,
            control_freq=100,  # control should happen fast enough so that simulation looks smooth
        )
    ))

# the noise objects for DDPG
#n_actions = env.action_space.shape[-1]
#sigma from DDPG paper
#action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))

if log:
  suff = str(time.strftime("_%d_%b_%Y_%H_%M_%S", time.localtime()))
else:
  suff = None

kwargs = {'verbose':2,
           'render':render,
           'param_noise':None,
           'action_noise':action_noise,
           'normalize_observations':normalize,
           'nb_train_steps':nb_train_steps,
           'nb_rollout_steps':nb_rollout_steps,
           'batch_size':batch_size,
           'critic_l2_reg':critic_l2_reg,
           'buffer_size':buffer_size,
           'random_exploration':random_exploration,
           'policy_kwargs':{'layer_norm':True},
           'logging':suff}
model = HER('MlpPolicy', env, DDPG, **kwargs)
start = time.time()

model.learn(total_timesteps=total_timesteps, log_interval=1)

if log :
  model.save("pkl/{}".format(suff))
  print("Saved as {0}, trained {1} primitive policy for {2} timesteps in {3}".format(suff, policy, total_timesteps, time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))

else:
  print("Trained {0} primitive policy for {1} timesteps in {2}".format(policy, total_timesteps, time.strftime('%H:%M:%S', time.gmtime(time.time()-start))))
