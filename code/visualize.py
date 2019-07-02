
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG

from sawyer_primitive_reach import SawyerPrimitiveReach

# use robosuite's gym_wrapper to wrap the sawyer stack env for baselines

# no object observation for PX,PY,PZ
# TODO turn on for pick policy?
env = GymWrapper(
        SawyerPrimitiveReach(
            prim_axis='x',
            has_renderer=True,
            ignore_done=True,
            has_offscreen_renderer=False,
      	    use_camera_obs=False,
            use_object_obs=True,
            control_freq=100,  # control should happen fast enough so that simulation looks smooth
        )
    )



model = DDPG.load("ddpg_test_x_batch1024.pkl")

for u in range(1):
  i = 0
  obs = env.reset()
#  while i != 1000:
  done = False
  while not done:
    i = i + 1
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

