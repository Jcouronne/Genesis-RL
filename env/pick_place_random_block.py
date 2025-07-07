import numpy as np
import genesis as gs
import torch
from .util import euler_to_quaternion
start_cube_pos = (0.65, 0.0, 0.02)
start_target_pos = (0, 0.65, 0.1)
cube_size = (0.04, 0.04, 0.04)  # size of the cube
target_size = (0.1, 0.1, 0.2)  # size of the target

class PickPlaceRandomBlockEnv:
    def __init__(self, vis, device, num_envs=1):
        self.device = device
        self.action_space = 5
        self.state_dim = 7
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.01,
            ),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
            ),
            show_viewer=vis,
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=cube_size,
                pos=start_cube_pos,
            )
        )
        self.target = self.scene.add_entity(
            gs.morphs.Box(
                size=target_size,
                pos=start_target_pos,
                fixed=True
            )
        )
        self.num_envs = num_envs
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)
        self.finger_pos = torch.full((self.num_envs, 2), 0.04, dtype=torch.float32, device=self.device) #test
        self.build_env()
    
    def build_env(self):
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        franka_pos = torch.tensor([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]).to(self.device)
        franka_pos = franka_pos.unsqueeze(0).repeat(self.num_envs, 1) 
        self.franka.set_qpos(franka_pos, envs_idx=self.envs_idx)
        self.scene.step()


        self.end_effector = self.franka.get_link("hand")
        ## here self.pos and self.quat is target for the end effector; not the cube. cube position is set in reset()
        pos = torch.tensor([0.65, 0.0, 0.135], dtype=torch.float32, device=self.device)

        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        #self.finger_pos = self.finger_pos.unsqueeze(0).repeat(self.num_envs, 1) #copy tensor along another dimension self.num_envs times
        quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos = self.pos,
            quat = self.quat,
        )
        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)

    def reset(self):
        self.build_env()
        ## random cube position
        cube_pos = np.array(start_cube_pos)
        R_min, R_max = 0.4, 0.7
        theta_min, theta_max = np.pi/10, -np.pi/10
        random_r = np.random.uniform(R_min, R_max, self.num_envs)
        random_theta = np.random.uniform(theta_min, theta_max, self.num_envs)
        random_x = random_r * np.cos(random_theta)
        random_y = random_r * np.sin(random_theta)
        cube_pos = np.column_stack((random_x, random_y, np.full(self.num_envs, cube_pos[2])))
        ## random cube orientation
        fixed_roll = 0
        fixed_pitch = 0
        random_yaws = np.random.uniform(0, 2 * np.pi, size=self.num_envs)
        quaternions = np.array([euler_to_quaternion(fixed_roll, fixed_pitch, yaw) for yaw in random_yaws])
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)
        self.cube.set_quat(quaternions, envs_idx=self.envs_idx)

        obs1 = self.cube.get_pos()
        obs2 = self.cube.get_quat()
        state = torch.concat([obs1, obs2], dim=1)
        return state


    def in_place_box(self, gripper_position, target_position, hitbox_range_xy=0.01):
        cube_pos = self.cube.get_pos()
        #Check if the cube position is within the target hitbox
        lower_bound_xy = target_position[:, :2] - hitbox_range_xy
        upper_bound_xy = target_position[:, :2] + hitbox_range_xy
        lower_bound_z = torch.tensor(cube_size[2]/3 + target_size[2], device=self.device)
        upper_bound_z = torch.tensor((2*cube_size[2])/3 + target_size[2], device=self.device)

        # Check if gripper is within the hitbox
        in_range_xy = torch.all((cube_pos[:, :2] >= lower_bound_xy) &
        (cube_pos[:, :2] <= upper_bound_xy), dim=1)
        in_range_z = (cube_pos[:, 2] >= lower_bound_z) & (cube_pos[:, 2] <= upper_bound_z)
        
        in_hitbox = in_range_xy & in_range_z

        return in_hitbox
        
    def step(self, actions):
        # Get positions and orientations for all environments
        block_position = self.cube.get_pos()
        block_quaternion = self.cube.get_quat()
        
        # Calculate offset based on cube and target sizes
        target_offset = torch.tensor([0.0, 0.0, cube_size[2]/2 + target_size[2]/2], device=self.device)
        target_position = self.target.get_pos() + target_offset
        end_effector = self.franka.get_link('hand')

        # Create action masks for all environments
        action_mask_0 = actions == 0  # Move above cube
        action_mask_1 = actions == 1  # Lift cube
        action_mask_2 = actions == 2  # Move above target
        action_mask_3 = actions == 3  # Place cube
        action_mask_4 = actions == 4  # End position

        # Initialize qpos tensor for all environments
        qpos = torch.zeros((self.num_envs, 9), device=self.device)
        pos = torch.zeros((self.num_envs, 3), device=self.device)

                # Set target positions based on action masks
        pos[action_mask_0] = block_position[action_mask_0] #- offset
        pos[action_mask_1] = block_position[action_mask_1] + torch.tensor([0.0, 0.0, 0.3], device=self.device)
        pos[action_mask_2] = target_position[action_mask_2] + torch.tensor([0.0, 0.0, 0.3], device=self.device)
        pos[action_mask_3] = target_position[action_mask_3]
        pos[action_mask_4] = target_position[action_mask_4] + torch.tensor([0.0, 0.0, 0.3], device=self.device)

        # Compute inverse kinematics for the target position
        target_qpos = self.franka.inverse_kinematics(
            link=end_effector,
            pos=pos.cpu().numpy(),
            quat=torch.tensor([0, 1, 0, 0], device=self.device).repeat(self.num_envs, 1).cpu().numpy(),
        )
        target_qpos = torch.tensor(target_qpos, device=self.device)

        # Update qpos for environments based on action masks
        qpos[action_mask_0] = target_qpos[action_mask_0]
        qpos[action_mask_1] = target_qpos[action_mask_1]
        qpos[action_mask_2] = target_qpos[action_mask_2]
        qpos[action_mask_3] = target_qpos[action_mask_3]
        qpos[action_mask_4] = target_qpos[action_mask_4]

        #control fingers first
        finger_positions = torch.full((self.num_envs, 2), 0.04, device=self.device)
        finger_positions[action_mask_0] = torch.tensor([0.4, 0.4], device=self.device)
        finger_positions[action_mask_1] = torch.tensor([0.0, 0.0], device=self.device)
        finger_positions[action_mask_2] = torch.tensor([0.0, 0.0], device=self.device)
        finger_positions[action_mask_3] = torch.tensor([0.0, 0.0], device=self.device)
        finger_positions[action_mask_4] = torch.tensor([0.4, 0.4], device=self.device)
        self.franka.control_dofs_position(finger_positions, self.fingers_dof, self.envs_idx)
        for i in range(10):
            self.scene.step()

        # Interpolate to the target position
        current_qpos = self.franka.get_qpos().cpu()

        for alpha in np.linspace(0, 1, num=100):  # 100 steps for interpolation
            interpolated_qpos = current_qpos + alpha * (qpos.cpu() - current_qpos)

            # Control DOFs for all environments (except fingers)
            self.franka.control_dofs_position(interpolated_qpos[:, :-2], self.motors_dof, self.envs_idx)
            self.scene.step()
        
        gripper_position = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2        
        states = torch.concat([block_position, block_quaternion], dim=1)    

        # Cube distance from target
        cdft = torch.norm(target_position - block_position, dim=1)

        # Combine rewards
        rewards = 1/(cdft + 1)*10
        dones = self.in_place_box(gripper_position, target_position) & torch.all(self.cube.get_vel() == 0,dim=1)
        return states, rewards, dones

if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32")
    env = PickPlaceRandomBlockEnv(vis=True, device="cuda")
