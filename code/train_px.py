
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG

# use robosuite's gym_wrapper to wrap the sawyer stack env for baselines

# no object observation for PX,PY,PZ
# TODO turn on for pick policy?
env = GymWrapper(
        suite.make(
            "SawyerStack",
            has_renderer=True,
            ignore_done=True,
            has_offscreen_renderer=False,
      	    use_camera_obs=False,
            use_object_obs=True,
            control_freq=100,  # control should happen fast enough so that simulation looks smooth
        )
    )

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=2, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=1000)
#model.save("ddpg_mountain")

#del model # remove to demonstrate saving and loading

#model = DDPG.load("ddpg_mountain")

obs = env.reset()
dones = True
i = 0
while i != 1000:
    i = i + 1
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

print("====={0}=====".format(i))
