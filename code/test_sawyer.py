from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.sawyer import SawyerEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import TableTopTask, UniformRandomSampler

class SawyerPrimitiveReach(SawyerEnv):
    """
    This class corresponds to the a primitive policy for the reach task on the Sawyer robot arm.

    Uniform random sample on x, y, or z axis in range of 20 to 60 cm for x and y, 0 to 40 cm for z (center of table)
    """

    def __init__(
        self,
        prim_axis = 'x',
        gripper_type="TwoFingerGripper",
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        placement_initializer=None,
        gripper_visualization=False,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
    ):
        """
        Args:
            prim_axis (str): which axis is being explored

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """

        self.prim_axis = prim_axis

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        # object placement initializer
        self.placement_initializer = placement_initializer
        limits = [.295, .305]
        if self.prim_axis == 'x':
            self.x_range = limits
            self.y_range = [0, 0]

        elif self.prim_axis == 'y':
            self.x_range = [0, 0]
            self.y_range = limits
        else:
            # TODO add z range in placement_initializer -> will cube fall??
            raise NotImplementedError

        super().__init__(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        self.mujoco_objects = OrderedDict([])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()
        pos_arr = np.array((0.56 + np.random.uniform(self.x_range[0], self.x_range[1]),
                            np.random.uniform(self.y_range[0], self.y_range[1]),
                            0.8))
        self.goal = pos_arr

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        # self.cube_body_id = self.sim.model.body_name2id("cube")
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]
        # self.cube_geom_id = self.sim.model.geom_name2id("cube")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # reset positions of objects
        # TODO adding marker reset
        pos_arr = np.array((0.56 + np.random.uniform(self.x_range[0], self.x_range[1]),
                            np.random.uniform(self.y_range[0], self.y_range[1]),
                            0.8))
        self.goal = pos_arr

        if self.has_renderer:
            self.viewer.viewer.add_marker(pos=pos_arr, size=np.array((0.02,0.02,0.02)), label='goal', rgba=[1, 0, 0, 0.5])
        self.model.place_objects()

        # reset joint positions
        init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

    def reward(self, action=None):
        """
        Reward function for the task.

        The dense reward has one component.

            Reaching: in [0, 1], to encourage the arm to reach the cube

        The sparse reward only consists of the lifting component.

        Args:
            action (np array): unused for this task

        Returns:
            reward (float): the reward
        """
     #   velocity_pen = np.linalg(np.array(
     #       [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
     #  ))
        velocity_pen = 0.0
        distance_threshold = 0.01
        cube_pos = self.goal
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        d = np.linalg.norm(gripper_site_pos - cube_pos)
        if self.reward_shaping: # dense
            reward = 1 - np.tanh(10 * d)
            if d <= distance_threshold:
                reward = 10.0
        else: # sparse (-1 or 0)
            reward = -np.float32(d > distance_threshold)
        
        return reward - velocity_pen

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
        # camera observations
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
            else:
                di["image"] = camera_obs

        # TODO change this to marker pos
        # low-level object information
        if self.use_object_obs:
            # position and rotation of object
            # cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
            cube_pos = self.goal
            # cube_quat = convert_quat(
            #     np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw"
            # )
            di["cube_pos"] = cube_pos
            # di["cube_quat"] = cube_quat

            gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["gripper_to_cube"] = gripper_site_pos - cube_pos

            # di["object-state"] = np.concatenate(
            #     [cube_pos, cube_quat, di["gripper_to_cube"]]
            # )
            di["object-state"] = np.concatenate(
                [cube_pos, di["gripper_to_cube"]]
            )

        return di

    # TODO change for marker
    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        # for contact in self.sim.data.contact[: self.sim.data.ncon]:
        #     if (
        #         self.sim.model.geom_id2name(contact.geom1)
        #         in self.gripper.contact_geoms()
        #         or self.sim.model.geom_id2name(contact.geom2)
        #         in self.gripper.contact_geoms()
        #     ):
        #         collision = True
        #         break
        EF_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])

        # within 2 cm
        if np.linalg.norm(self.goal-EF_pos) <= 0.02:
            collision = True
        return collision

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        return self._check_contact()

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to cube
        if self.gripper_visualization:
            # get distance to cube
            cube_site_id = self.sim.model.site_name2id("cube")
            dist = np.sum(
                np.square(
                    self.sim.data.site_xpos[cube_site_id]
                    - self.sim.data.get_site_xpos("grip_site")
                )
            )

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba
