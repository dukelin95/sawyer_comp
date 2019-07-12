import time
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper

from gym_goal_wrapper import GymGoalEnvWrapper
from ik_wrapper import IKWrapper
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG
from stable_baselines import HER

from utils import HERGoalEnvWrapper
# from sawyer_primitive_reach import SawyerPrimitiveReach
from test_sawyer import SawyerPrimitiveReach
import argparse


render = True

policy = 'x'
limits = [0.2, 0.2] 
table_full_size = (0.8, 0.8, 0.4)

nb_train_steps = 25
nb_rollout_steps = 50
batch_size = 64
critic_l2_reg = 0.01
buffer_size=int(1e6)
normalize = True

action_noise=None
total_timesteps = int(5e3)

env1 = SawyerPrimitiveReach(
            prim_axis=policy,
            limits=limits,
            table_full_size=table_full_size,
            has_renderer=render,
            has_offscreen_renderer=False,
      	    use_camera_obs=False,
            use_object_obs=True,
            horizon = 500,
            control_freq=100,  # control should happen fast enough so that simulation looks smooth
        )
env2 = IKWrapper(env1)
env3 = GymGoalEnvWrapper(env2)
env3.reset()
env3.render()

def find(env1, env3, loop):
    p = []
    q = []
    for i in range(loop):
        o = env3.reset()
        env3.render()
        p.append(np.array(env1.sim.data.site_xpos[env1.eef_site_id]))
        q.append(suite.utils.transform_utils.convert_quat(
                 env1.sim.data.get_body_xquat("right_hand"), to='xyzw'))
    return p, q

def view(env, loop, action=None):
    for i in range(loop):
        if action is None:
           action = np.array([0.00, 0.00, 0.0])
           action[np.random.randint(3)] = 0.01
        obs_dict, r, d, i = env.step(action)
        env3.viewer.viewer.add_marker(pos=env3.goal, size=np.array((0.02,0.02,0.02)), label='goal', rgba=[1, 0, 0, 0.5])
        print(action)
        env.render()
