import numpy as np
import genesis as gs
import torch
from .util import euler_to_quaternion
start_cube_pos = (0.65, 0.0, 0.02)
start_target_pos = (0, 0.65, 0.15)
cube_size = (0.04, 0.04, 0.04)
target_size = (0.1, 0.1, 0.3)

class ComplexPickPlaceRandomBlockEnv:
    def __init__(self, vis, device, num_envs=1):
        self.device = device
        self.action_space = 5
        self.state_dim = 9
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=100,
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
        self.previous_actions = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.build_env()
    
    def build_env(self):
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        franka_pos = torch.tensor([-1.0175,  1.2663,  1.7025, -1.6078, -1.2595,  1.4565,  1.4139,  0.0400, 0.0400], 
                                  dtype=torch.float64, device=self.device)
        franka_pos = franka_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self.franka.set_qpos(franka_pos, envs_idx=self.envs_idx)
        self.scene.step()

        self.end_effector = self.franka.get_link("hand")
        ## here self.pos and self.quat is target for the end effector; not the cube. cube position is set in reset()
        pos = torch.tensor([0.65, 0.0, 0.4], dtype=torch.float64, device=self.device)

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
        R_min, R_max = 0.5, 0.7
        ## random cube position
        cube_pos = np.array(start_cube_pos)
        cube_theta_min, cube_theta_max = np.pi/5, -np.pi/5
        cube_random_r = np.random.uniform(R_min, R_max, self.num_envs)
        cube_random_theta = np.random.uniform(cube_theta_min, cube_theta_max, self.num_envs)
        cube_random_x = cube_random_r * np.cos(cube_random_theta)
        cube_random_y = cube_random_r * np.sin(cube_random_theta)
        cube_pos = np.column_stack((cube_random_x, cube_random_y, np.full(self.num_envs, cube_pos[2])))

        quaternions = torch.tensor([1, 0, 0, 0], device=self.device, dtype=torch.float64).unsqueeze(0).repeat(self.num_envs, 1)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)
        self.cube.set_quat(quaternions, envs_idx=self.envs_idx)

        obs1 = self.cube.get_pos().to(dtype=torch.float64)
        obs2 = self.target.get_pos().to(dtype=torch.float64)
        obs3 = (self.franka.get_link("left_finger").get_pos().to(dtype=torch.float64) + self.franka.get_link("right_finger").get_pos().to(dtype=torch.float64)) / 2   
        state = torch.concat([obs1,obs2,obs3], dim=1)
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
        
    def step(self, actions):
        target_position = self.target.get_pos()
        target_offset = torch.tensor([0.0, 0.0, (target_size[2]/2)+cube_size[2]/2], device=self.device, dtype=torch.float64)
        hover_offset = torch.tensor([0.0, 0.0, 0.3], device=self.device, dtype=torch.float64)
        
        # action_mask_0 = actions == 0  # Open gripper
        action_mask_1 = actions == 1  # Close gripper
        action_mask_2 = actions == 2  # Lift gripper
        action_mask_3 = actions == 3  # Lower gripper
        action_mask_4 = actions == 4  # Move left
        action_mask_5 = actions == 5  # Move right
        action_mask_6 = actions == 6  # Move forward
        action_mask_7 = actions == 7  # Move backward

        finger_pos = torch.full(
            (self.num_envs, 2), 0.04, dtype=torch.float32, device=self.device
        )
        finger_pos[action_mask_1] = 0
        finger_pos[action_mask_2] = 0

        pos = self.pos.clone()
        pos[action_mask_2, 2] = 0.4
        pos[action_mask_3, 2] = 0
        pos[action_mask_4, 0] -= 0.05
        pos[action_mask_5, 0] += 0.05
        pos[action_mask_6, 1] -= 0.05
        pos[action_mask_7, 1] += 0.05

        self.pos = pos
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=pos,
            quat=self.quat,
        )

        self.franka.control_dofs_position(
            self.qpos[:, :-2], self.motors_dof, self.envs_idx
        )
        self.franka.control_dofs_position(finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        block_position = self.cube.get_pos()
        gripper_position = (
            self.franka.get_link("left_finger").get_pos()
            + self.franka.get_link("right_finger").get_pos()
        ) / 2

        # Cube distance from target
        cdft = torch.norm((target_position + target_offset) - block_position, dim=1)
        # Effector distance from end position
        #edfs = torch.norm(gripper_position - (target_position + hover_offset), dim=1)
        # Get cube velocity and calculate its norm for each environment
        cube_velocity = torch.norm(self.cube.get_vel().to(dtype=torch.float64), dim=1)  # Shape is [num_envs, 3]

        # Create a matching zero tensor
        zero_vel = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        # Check if cube is at rest by comparing velocity norms
        cube_at_rest = torch.isclose(cube_velocity, zero_vel, atol=0.01)

        states = torch.concat(
                    [
                        block_position,
                        self.franka.get_link("left_finger").get_pos(),
                        self.franka.get_link("right_finger").get_pos(),
                    ],
                    dim=1,
                )

        # Calculate distance from gripper to start position for environments that are done
        rewards = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        rewards = 1/(cdft + 1)
        dones = self.in_place_box(target_position) & cube_at_rest
        return states, rewards, dones
    
if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="64")
    env = ComplexPickPlaceRandomBlockEnv(vis=True, device="cuda")
