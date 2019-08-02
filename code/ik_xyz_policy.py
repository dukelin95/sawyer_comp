import numpy as np

from robosuite.controllers import SawyerIKController

from sandbox.rocky.tf.policies.base import Policy

class IK_Policy(Policy):

    def __init__(self, axis='x'):
        # TODO is this env_spec needed?
        super().init(env_spec = None)
        if axis == 'x':
            self.axis = 0
        elif axis == 'y':
            self.axis = 1
        elif axis == 'z':
            self.axis = 2
        else:
            raise ValueError("Only x, y, or z accepted")

    def get_action(self, obs):
        # order: gripper_qpos, eef_pos, marker/cube_pos
        marker_pos = obs[5:8]
        eef_pos = obs[2:5]
        diff = marker_pos - eef_pos
        action = np.zeros(3)
        action[self.axis] = diff[self.axis]
        return action

