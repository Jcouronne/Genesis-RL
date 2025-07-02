import numpy as np
import genesis as gs
import torch

class WaterFrankaEnv:
    def __init__(self, vis, device, num_envs=1):
        self.device = device
        self.action_space = 8  
        self.state_dim = 3  
        assert num_envs == 1, "Liquid only supports num_envs=1"
        self.num_envs = num_envs
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=4e-3,
                substeps=10,
            ),
            sph_options=gs.options.SPHOptions(
                lower_bound=(0.5, -0.15, 0.0),
                upper_bound=(0.8, 0.15, 10.5),
                particle_size=0.01,
            ),
            vis_options=gs.options.VisOptions(
                visualize_sph_boundary=True,
            ),
            show_viewer=vis,
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.franka = self.scene.add_entity(
            # gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"),
            gs.morphs.URDF(file="../assets/urdf/panda_bullet/panda.urdf", fixed=True), 
        )
        self.liquid = self.scene.add_entity(
            material=gs.materials.SPH.Liquid(),
            morph=gs.morphs.Box(
                pos=(0.65, 0.0, 0.25),
                size=(0.3, 0.3, 0.5),
            ),
            surface=gs.surfaces.Default(
                color=(0.0, 0.0, 1.0),
                vis_mode="particle",  # or "recon"
            ),
        )
        """
        print(dir(gs.materials.SPH.Liquid()))
        ['__class__', '__colorized__repr__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_data_oriented', '_exponent', '_gamma', '_is_debugger', '_mu', '_repr_brief', '_repr_briefer', '_repr_type', '_rho', '_sampler', '_stiffness', '_uid', 'exponent', 'gamma', 'mu', 'rho', 'sampler', 'stiffness', 'uid']
        print(dir(self.liquid))
        ['__class__', '__colorized__repr__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_add_to_solver', '_add_to_solver_', '_add_vverts_to_solver', '_assert_active', '_ckpt', '_data_oriented', '_idx', '_init_particles_offset', '_is_debugger', '_kernel_get_mass', '_kernel_get_particles', '_material', '_morph', '_n_particles', '_need_skinning', '_particle_size', '_particle_start', '_particles', '_queried_states', '_repr_brief', '_repr_briefer', '_repr_type', '_scene', '_sim', '_solver', '_surface', '_tgt', '_tgt_buffer', '_tgt_keys', '_uid', '_vface_start', '_vfaces', '_vmesh', '_vvert_start', '_vverts', 'activate', 'active', 'add_grad_from_state', 'clear_grad', 'collect_output_grads', 'deactivate', 'get_frame', 'get_mass', 'get_particles', 'get_state', 'idx', 'init_ckpt', 'init_particles', 'init_sampler', 'init_tgt_keys', 'init_tgt_vars', 'is_built', 'load_ckpt', 'material', 'morph', 'n_particles', 'n_vfaces', 'n_vverts', 'particle_end', 'particle_size', 'particle_start', 'process_input', 'process_input_grad', 'reset_grad', 'sample', 'sampler', 'save_ckpt', 'scene', 'set_active', 'set_pos', 'set_pos_grad', 'set_position', 'set_vel', 'set_vel_grad', 'set_velocity', 'sim', 'solver', 'surface', 'uid', 'vface_end', 'vface_start', 'vmesh', 'vvert_end', 'vvert_start']
        print(self.liquid.n_particles)
        print(self.liquid.get_state().pos) # same as: print(self.liquid.get_particles())
        """
        self.scene.build()
        self.build_env()
    
    def build_env(self):
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)

        self.end_effector = self.franka.get_link("panda_leftfinger")

        self.pos = torch.tensor([0.65, 0.0, 0.135], dtype=torch.float32, device=self.device)
        self.quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos = self.pos,
            quat = self.quat,
        )

        franka_lower_limits, franka_upper_limits = self.franka.get_dofs_limit()
        self.qpos = 0.7 * (franka_upper_limits + franka_lower_limits)

        self.franka.set_qpos(self.qpos)
        self.scene.step()
        self.franka.control_dofs_position(self.qpos[:-2], self.motors_dof)

    def reset(self):
        self.build_env()
        gripper_position = (self.franka.get_link("panda_leftfinger").get_pos() + self.franka.get_link("panda_rightfinger").get_pos()) / 2
        states = torch.tensor(gripper_position, device=self.device).unsqueeze(0)
        return states

    def step(self, actions):
        if actions is None:
            self.scene.step()
            return None
        finger_pos = torch.tensor([0.04, 0.04], device=self.device)
        pos = self.pos.clone()
        if actions == 1: # Close gripper
            finger_pos[:] = 0
        elif actions == 2: # Lift gripper 
            finger_pos[:] = 0
            pos[2] = 1.0
        elif actions == 3: # Lower gripper
            pos[2] = 0
        elif actions == 4: # Move left
            pos[0] -= 0.05
        elif actions == 5: # Move right
            pos[0] += 0.05
        elif actions == 6: # Move forward
            pos[1] -= 0.05
        elif actions == 7: # Move backward
            pos[1] += 0.05

        self.pos = pos
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=pos,
            quat=self.quat,
        )

        self.franka.control_dofs_position(self.qpos[:-2], self.motors_dof)
        self.franka.control_dofs_position(finger_pos, self.fingers_dof)
        self.scene.step()

        gripper_position = (self.franka.get_link("panda_leftfinger").get_pos() + self.franka.get_link("panda_rightfinger").get_pos()) / 2
        states = torch.tensor(gripper_position, device=self.device).unsqueeze(0)

        rewards = self.liquid.get_particles()[:, 2].max() 
        dones = False
        rewards = torch.tensor([rewards], device=self.device)
        dones = torch.tensor([dones], device=self.device)
        return states, rewards, dones

if __name__ == "__main__":
    gs.init()
    env = WaterFrankaEnv(vis=True, device=gs.device, num_envs=1)
    horizon = 10000
    up = False
    for i in range(horizon):
        if i % 200 == 0:
            up = not up
        if i < 100:
            actions = None
        else:
            if up:
                actions = torch.ones(1, device="cuda:0") * 2
            else:
                actions = torch.ones(1, device="cuda:0") * 3
        env.step(actions)