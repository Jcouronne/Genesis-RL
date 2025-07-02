import numpy as np
import genesis as gs
import torch

class GraspFixedRodEnv:
    def __init__(self, vis, device, num_envs=1):
        self.device = device
        self.action_space = 8  
        self.state_dim = 6  

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
                size=(0.2, 0.04, 0.04), # long rod
                pos=(0.65, 0.0, 0.02),
            )
        )
        self.num_envs = num_envs
        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)
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
        # fixed cube position
        cube_pos = np.array([0.65, 0.0, 0.02])
        cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)

        obs1 = self.cube.get_pos()
        obs2 = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2 
        state = torch.concat([obs1, obs2], dim=1)
        return state

    def step(self, actions):
        action_mask_0 = actions == 0 # Open gripper
        action_mask_1 = actions == 1 # Close gripper
        action_mask_2 = actions == 2 # Lift gripper
        action_mask_3 = actions == 3 # Lower gripper
        action_mask_4 = actions == 4 # Move left
        action_mask_5 = actions == 5 # Move right
        action_mask_6 = actions == 6 # Move forward
        action_mask_7 = actions == 7 # Move backward

        finger_pos = torch.full((self.num_envs, 2), 0.04, dtype=torch.float32, device=self.device)
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

        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()

        block_position = self.cube.get_pos()
        gripper_position = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2
        states = torch.concat([block_position, gripper_position], dim=1)

        rewards = -torch.norm(block_position - gripper_position, dim=1) + torch.maximum(torch.tensor(0.02), block_position[:, 2]) * 10
        dones = block_position[:, 2] > 0.35
        return states, rewards, dones


if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32")
    env = GraspFixedRodEnv(vis=True)