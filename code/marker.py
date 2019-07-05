from test_sawyer import SawyerPrimitiveReach
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.environments.sawyer_nut_assembly import SawyerNutAssembly
import numpy as np
env = GymWrapper(
        SawyerPrimitiveReach(
            prim_axis='x',
#	SawyerNutAssembly(
            has_offscreen_renderer=False,
            use_indicator_object=True,
            has_renderer=True,
      	    use_camera_obs=False,
            use_object_obs=True,
            horizon = 500,
            control_freq=100,  # control should happen fast enough so that simulation looks smooth
        )
    )

obs = env.reset()
env.render()
for i in range(1000):
  # env.viewer.viewer.add_marker(size=np.array([0.02,0.02,0.02]),pos=np.array([0.5,0,1]), label='dest')
  action = np.random.randn(env.dof)
  o, r, d, i = env.step(action)
  env.render()
