import numpy as np
import genesis as gs
import torch
#Stability test :
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


start_cube_pos = (0.65, 0.0, 0.02)

class GraspFixedBlockEnv:
    def __init__(self, vis, device, num_envs=1):
        self.device = device
        self.action_space = 8  
        self.state_dim = 13
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
                size=(0.04, 0.04, 0.04), # block
                pos=start_cube_pos,
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
        # fixed cube position
        cube_pos = np.array(start_cube_pos)
        cube_pos = np.repeat(cube_pos[np.newaxis], self.num_envs, axis=0)
        self.cube.set_pos(cube_pos, envs_idx=self.envs_idx)

        obs1 = self.cube.get_pos()
        obs2 = self.cube.get_quat()
        #obs2 = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2 
        obs3 = self.franka.get_link("left_finger").get_pos()
        obs4 = self.franka.get_link("right_finger").get_pos()
        state = torch.concat([obs1, obs2, obs3, obs4], dim=1)
        return state

#    def in_pick_up_box(self, gripper_position, block_position, hitbox_range_xy=0.1, hitbox_height=0.2):
#        """
#        Check if the gripper position is within a hitbox defined above the block position.
#        """
#        cube_position = torch.tensor(start_cube_pos)
#        # Calculate the boundaries of the hitbox
#        lower_bound_xy = start_cube_pos[:, :2] - hitbox_range_xy
#        upper_bound_xy = start_cube_pos[:, :2] + hitbox_range_xy
#        lower_bound_z = start_cube_pos[:, 2]
#        upper_bound_z = start_cube_pos[:, 2] + hitbox_height

        # Check if gripper is within the hitbox
#        in_range_xy = torch.all((gripper_position[:, :2] >= lower_bound_xy) &
#        (gripper_position[:, :2] <= upper_bound_xy), dim=1)
#        in_range_z = (gripper_position[:, 2] >= lower_bound_z) & (gripper_position[:, 2] <= upper_bound_z)
#        in_hitbox = in_range_xy & in_range_z

#        return in_hitbox
        
    def step(self, actions):
        action_mask_0 = actions == 0 # Open gripper
        action_mask_1 = actions == 1 # Close gripper
        action_mask_2 = actions == 2 # Move up
        action_mask_3 = actions == 3 # Move down
        action_mask_4 = actions == 4 # Move left
        action_mask_5 = actions == 5 # Move right
        action_mask_6 = actions == 6 # Move forward
        action_mask_7 = actions == 7 # Move backward
        
        #finger_pos = self.finger_pos.clone()
        #finger_pos = torch.full((self.num_envs, 2), 0.04, dtype=torch.float32, device=self.device)
        self.finger_pos[action_mask_0] = 0.04
        self.finger_pos[action_mask_1] = 0
        
        pos = self.pos.clone()
        pos[action_mask_2, 2] += 0.2
        pos[action_mask_3, 2] -= 0.01
        pos[action_mask_4, 0] -= 0.01
        pos[action_mask_5, 0] += 0.01
        pos[action_mask_6, 1] -= 0.01
        pos[action_mask_7, 1] += 0.01

        self.pos = pos
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=pos,
            quat=self.quat,
        )
        
        self.franka.control_dofs_position(self.qpos[:, :-2], self.motors_dof, self.envs_idx)
        self.franka.control_dofs_position(self.finger_pos, self.fingers_dof, self.envs_idx)
        self.scene.step()
        
        block_position = self.cube.get_pos()
        block_quaternion = self.cube.get_quat()
        gripper_position = (self.franka.get_link("left_finger").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2
        gripper1_position = self.franka.get_link("left_finger").get_pos()
        gripper2_position = self.franka.get_link("right_finger").get_pos()
        
        states = torch.concat([block_position, gripper1_position, gripper2_position, block_quaternion], dim=1)    

        # -Effector distance from the cube
        dee = torch.norm(block_position - gripper_position, dim=1)
                                
        # -Finger distance from the cube
        dfg = torch.norm(block_position - gripper1_position, dim=1) + torch.norm(block_position - gripper2_position, dim=1)
        
        # +Height of the cube
        #height_reward = block_position[:, 2]
        height_reward = torch.where(
            block_position[:, 2] > 0.4,
            torch.abs(block_position[:, 2] - 0.35)*1/2,
            (block_position[:, 2] - 0.0199)*10
        )
        # +Being aligned with the cube in the pick-up box
        #norm_penalty = torch.norm(start_cube_pos[:2] - gripper_position[:, :2], dim=1)

        # Combine rewards
        rewards = 1/(dee + dfg*5 + 1) + height_reward
        #rewards[in_box] -= norm_penalty[in_box]

        #rewards = 1/torch.exp(+torch.norm(block_position - gripper1_position, dim=1) + torch.norm(block_position - gripper2_position, dim=1))*10 + 0.0199*2
        #+ (block_position[:, 2]-0.0199)*100 if in_pick_up_box(): -torch.norm(start_cube_pos - gripper_position, dim=1)
        
        #rewards = -torch.norm(block_position - gripper_position, dim=1)+ torch.maximum(torch.tensor(0.02), (block_position[:, 2]-0.2)) * 10 #default reward
        dones = block_position[:, 2] > 0.35
        return states, rewards, dones

if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32")
    env = GraspFixedBlockEnv(vis=True)
