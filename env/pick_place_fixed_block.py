import numpy as np
import genesis as gs
import torch
start_cube_pos = (0.65, 0.0, 0.02)
start_target_pos = (0, 0.65, 0.1)
cube_size = (0.04, 0.04, 0.04)  # size of the cube
target_size = (0.1, 0.1, 0.2)  # size of the target

class PickPlaceFixedBlockEnv:
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
        self.build_env()
    
    def build_env(self):
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        franka_pos = torch.tensor([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04], 
                                  dtype=torch.float64, device=self.device)
        franka_pos = franka_pos.unsqueeze(0).repeat(self.num_envs, 1) 
        self.franka.set_qpos(franka_pos, envs_idx=self.envs_idx)
        self.scene.step()

        self.end_effector = self.franka.get_link("hand")
        ## here self.pos and self.quat is target for the end effector; not the cube. cube position is set in reset()
        pos = torch.tensor([0.65, 0.0, 0.135], dtype=torch.float64, device=self.device)

        self.pos = pos.unsqueeze(0).repeat(self.num_envs, 1)
        quat = torch.tensor([0, 1, 0, 0], dtype=torch.float64, device=self.device)
        self.quat = quat.unsqueeze(0).repeat(self.num_envs, 1)
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos = self.pos,
            quat = self.quat,
        )
        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)

    def reset(self):
        self.build_env()
        # fixed cube position
        cube_pos = np.array(start_cube_pos)
        cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)

        obs1 = self.cube.get_pos().to(dtype=torch.float64)
        obs2 = self.cube.get_quat().to(dtype=torch.float64)
        state = torch.concat([obs1, obs2], dim=1)
        return state
        
    def in_place_box(self, target_position, hitbox_range_xy=0.02):
        cube_pos = self.cube.get_pos().to(dtype=torch.float64)
        # Check if the cube position is within the target hitbox
        lower_bound_xy = target_position[:, :2] - hitbox_range_xy
        upper_bound_xy = target_position[:, :2] + hitbox_range_xy
        
        # Fix the z bounds with proper tensor shapes
        lower_bound_z = torch.full((self.num_envs,), target_size[2], device=self.device, dtype=torch.float64)
        upper_bound_z = torch.full((self.num_envs,), cube_size[2] + target_size[2], device=self.device, dtype=torch.float64)

        in_range_xy = torch.all((cube_pos[:, :2] >= lower_bound_xy) &
                              (cube_pos[:, :2] <= upper_bound_xy), dim=1)
        in_range_z = (cube_pos[:, 2] >= lower_bound_z) & (cube_pos[:, 2] <= upper_bound_z)
        
        in_hitbox = in_range_xy & in_range_z
        return in_hitbox

    def in_end_pos(self, start_target_pos, hitbox_range_xy=0.02, hitbox_z_height=0.1):
        effector_pos = self.end_effector.get_pos().to(dtype=torch.float64)
        # Check if the cube position is within the target hitbox
        lower_bound_xy = start_target_pos[:, :2] - hitbox_range_xy
        upper_bound_xy = start_target_pos[:, :2] + hitbox_range_xy
        
        # Fix the z bounds with proper tensor shapes
        lower_bound_z = start_target_pos[:, 2] + 0.4
        upper_bound_z = start_target_pos[:, 2] + hitbox_z_height 

        in_range_xy = torch.all((effector_pos[:, :2] >= lower_bound_xy) &
                              (effector_pos[:, :2] <= upper_bound_xy), dim=1)
        in_range_z = (effector_pos[:, 2] >= lower_bound_z) & (effector_pos[:, 2] <= upper_bound_z)
        
        in_hitbox = in_range_xy & in_range_z
        return in_hitbox
        
    def step(self, actions, start_target_pos=start_target_pos):
        # Get positions and orientations for all environments
        block_position = self.cube.get_pos().to(dtype=torch.float64)
        block_quaternion = self.cube.get_quat().to(dtype=torch.float64)
        target_position = torch.tensor(start_target_pos, device=self.device, dtype=torch.float64).unsqueeze(0).repeat(self.num_envs, 1)
        
        # Calculate offset based on cube and target sizes
        end_effector_offset = torch.tensor([-0.05, -0.05, 0.1], device=self.device, dtype=torch.float64) + cube_size[2]
        target_offset = torch.tensor([0.0, 0.0, target_size[2]/2], device=self.device, dtype=torch.float64)
        end_effector = self.franka.get_link('hand')
        hover_offset = torch.tensor([0.0, 0.0, 0.3], device=self.device, dtype=torch.float64)

        # Create action masks for all environments
        action_mask_0 = actions == 0  # Move above cube
        action_mask_1 = actions == 1  # Lift cube
        action_mask_2 = actions == 2  # Move above target
        action_mask_3 = actions == 3  # Place cube
        action_mask_4 = actions == 4  # End position

        # Initialize qpos tensor for all environments
        qpos = torch.zeros((self.num_envs, 9), device=self.device, dtype=torch.float64)
        pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float64)

        # Set target positions based on action masks - using target_position from above
        pos[action_mask_0] = block_position[action_mask_0]
        pos[action_mask_1] = block_position[action_mask_1] + hover_offset
        pos[action_mask_2] = target_position[action_mask_2] + hover_offset
        pos[action_mask_3] = target_position[action_mask_3] + target_offset
        pos[action_mask_4] = target_position[action_mask_4] + hover_offset
        # Compute inverse kinematics for the target position
        target_qpos = self.franka.inverse_kinematics(
            link=end_effector,
            pos=(pos + end_effector_offset).cpu().numpy(),
            quat=torch.tensor([0, 1, 0, 0], device=self.device, dtype=torch.float64).repeat(self.num_envs, 1).cpu().numpy(),
        )
        target_qpos = torch.tensor(target_qpos, device=self.device, dtype=torch.float64)

        # Update qpos for environments based on action masks
        qpos[action_mask_0] = target_qpos[action_mask_0]
        qpos[action_mask_1] = target_qpos[action_mask_1]
        qpos[action_mask_2] = target_qpos[action_mask_2]
        qpos[action_mask_3] = target_qpos[action_mask_3]
        qpos[action_mask_4] = target_qpos[action_mask_4]

        #control fingers first
        finger_positions = torch.full((self.num_envs, 2), 0.04, device=self.device, dtype=torch.float64)
        finger_positions[action_mask_0] = torch.tensor([0.4, 0.4], device=self.device, dtype=torch.float64)
        finger_positions[action_mask_1] = torch.tensor([0.0, 0.0], device=self.device, dtype=torch.float64)
        finger_positions[action_mask_2] = torch.tensor([0.0, 0.0], device=self.device, dtype=torch.float64)
        finger_positions[action_mask_3] = torch.tensor([0.0, 0.0], device=self.device, dtype=torch.float64)
        finger_positions[action_mask_4] = torch.tensor([0.4, 0.4], device=self.device, dtype=torch.float64)
        self.franka.control_dofs_position(finger_positions, self.fingers_dof, self.envs_idx)
        for i in range(50):
            self.scene.step()

        # Interpolate to the target position
        current_qpos = self.franka.get_qpos().cpu().to(dtype=torch.float64)

        for alpha in np.linspace(0, 1, num=200):  # num steps for interpolation (also speed)
            interpolated_qpos = current_qpos + alpha * (qpos.cpu() - current_qpos)

            # Control DOFs for all environments (except fingers)
            self.franka.control_dofs_position(interpolated_qpos[:, :-2], self.motors_dof, self.envs_idx)
            self.scene.step()
        
        gripper_position = (self.franka.get_link("left_finger").get_pos().to(dtype=torch.float64) + 
                            self.franka.get_link("right_finger").get_pos().to(dtype=torch.float64)) / 2        
        states = torch.concat([block_position, block_quaternion], dim=1)    

        start_target_pos = torch.tensor(start_target_pos, device=self.device, dtype=torch.float64).unsqueeze(0).repeat(self.num_envs, 1)
        # Cube distance from target
        cdft = torch.norm(target_position - block_position, dim=1)
        # Cube distance from effector
        cdfe = torch.norm(gripper_position - block_position, dim=1)
        # Effector distance from end position
        edfs = torch.norm(gripper_position - (start_target_pos + hover_offset), dim=1)
        # Get cube velocity and calculate its norm for each environment
        cube_velocity = torch.norm(self.cube.get_vel().to(dtype=torch.float64), dim=1)  # Shape is [num_envs, 3]

        # Create a matching zero tensor
        zero_vel = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        # Check if cube is at rest by comparing velocity norms
        cube_at_rest = torch.isclose(cube_velocity, zero_vel, atol=0.01)
        
        # Calculate distance from gripper to start position for environments that are done
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        for env_idx in range(self.num_envs):
            if (self.in_place_box(target_position) & cube_at_rest)[env_idx]:
                print(f"edfs for env {env_idx}: {edfs[env_idx]}")
                rewards[env_idx] = 1/(edfs[env_idx]*10 + cdft[env_idx]*2 + 1)*100 + cube_at_rest[env_idx]*20
                print(f"Reward for env {env_idx}: {rewards[env_idx]}")
            else:
                rewards[env_idx] = 1/(cdft[env_idx] + 1)+cdfe[env_idx]

        dones = self.in_place_box(target_position) & cube_at_rest & self.in_end_pos(start_target_pos)
        return states, rewards, dones
    
if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="64")  # Changed to 64-bit precision
    env = PickPlaceFixedBlockEnv(vis=True, device="cuda")
