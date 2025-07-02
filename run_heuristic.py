import argparse
import numpy as np
import time
import genesis as gs
import torch

gs.init(backend=gs.gpu, precision="32")

class Env:
    def __init__(self, vis, num_envs=1):
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
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04), # block
                # size=(0.4, 0.04, 0.04), # long rod
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

class CheckSuccess:
    def __init__(self):
        self.time = 0

    def is_block_in_gripper(self, block_position, gripper_position, threshold=0.02):
        distance = torch.norm(block_position - gripper_position)
        return distance < threshold

    def check_success(self, block_position, gripper_position):
        if block_position[2] > 0.15 and self.is_block_in_gripper(block_position, gripper_position, 0.5):
            self.time += 1
            print("Block successfully lifted!")
        return self.time > 10

class Rollout:
    def __init__(self, args):
        self.finger_pos = 0.0
        self.env = Env(args.vis)
        self.checker = CheckSuccess()
    
    def rollout(self):
        for i in range(20):
            print("grasp", i)
            self.env.franka.control_dofs_position(self.env.qpos[:-2], self.env.motors_dof)
            self.env.franka.control_dofs_position(np.array([self.finger_pos, self.finger_pos]), self.env.fingers_dof)
            self.env.scene.step()

            block_position = self.env.cube.get_pos() 
            gripper_position = (self.env.franka.get_link("left_finger").get_pos() \
                                + self.env.franka.get_link("right_finger").get_pos()) / 2
            
            if self.checker.is_block_in_gripper(block_position, gripper_position, 0.051):
                print("Block successfully picked up!")
                break
        else:
            print("Failed to pick up the block within 20 attempts.")

        # lift
        qpos_lift = self.env.franka.inverse_kinematics(
            link=self.env.end_effector,
            pos=np.array([0.65, 0.0, 0.3]),
            quat=np.array([0, 1, 0, 0]),
        )

        for i in range(100):
            print("lift", i)
            self.env.franka.control_dofs_position(qpos_lift[:-2], self.env.motors_dof)
            self.env.franka.control_dofs_position(np.array([self.finger_pos, self.finger_pos]), self.env.fingers_dof)
            self.env.scene.step()

            block_position_after_lift = self.env.cube.get_pos()  
            gripper_position = (self.env.franka.get_link("left_finger").get_pos() + self.env.franka.get_link("right_finger").get_pos()) / 2
            if self.checker.check_success(block_position_after_lift, gripper_position):
                break
        else:
            print("Failed to lift the block within 100 steps.")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    rollout = Rollout(args)
    rollout.rollout()
