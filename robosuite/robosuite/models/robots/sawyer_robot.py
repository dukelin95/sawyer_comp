import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string

from robosuite.controllers import SawyerIKController
import robosuite
import os

class Sawyer(Robot):
    """Sawyer is a witty single-arm robot designed by Rethink Robotics."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/sawyer/robot.xml"))

        self.bottom_offset = np.array([0, 0, -0.913])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 7

    @property
    def joints(self):
        return ["right_j{}".format(x) for x in range(7)]

    @property
    def init_qpos(self):
        # return a random initialization

        constant_quat = np.array([-0.01704371, -0.99972409,  0.00199679, -0.01603944])
        target_position = np.array([0.58038172, -0.01562932,  0.90211762]) \
                         + np.random.uniform(-0.2, 0.2, 3)

        self.controller = SawyerIKController(
            bullet_data_path=os.path.join(robosuite.models.assets_root, "bullet_data"),
            robot_jpos_getter=self._robot_jpos_getter,
        )
        joint_list = self.controller.inverse_kinematics(target_position, constant_quat)
        return np.array(joint_list)
        # return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])

    # helper function for ik controller
    def _robot_jpos_getter(self):
        return np.array(self.joints())