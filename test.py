import numpy as np
import genesis as gs
import time

import torch

# Constants
start_cube_pos = (0.65, 0.0, 0.02)
start_target_pos = (0, 0.65, 0)
cube_size = (0.04, 0.04, 0.04)  # size of the cube
target_size = (0.1, 0.1, 0.5)  # size of the target

target_position = torch.tensor(start_target_pos, device="cuda", dtype=torch.float64)
target_offset = torch.tensor([0.0, 0.0, (target_size[2]/2)+], device="cuda", dtype=torch.float64)

start_cube_pos = (0.65, 0.0, 0.02)
########################## init ##########################
gs.init(backend=gs.gpu, precision="32")

scene = gs.Scene(
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
    show_viewer=True,
)

# Add entities to the scene
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"))
cube = scene.add_entity(gs.morphs.Box(size=cube_size, pos=start_cube_pos))
target = scene.add_entity(gs.morphs.Box(size=target_size, pos=start_target_pos, fixed=True))

########################## build ##########################
scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# Set control gains for the Franka robot
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
)

# Get the end-effector link
end_effector = franka.get_link('hand')


# Calculate target position with offset
target_pos = np.array(start_target_pos) + np.array([0.0, 0.0, cube_size[2] + target_size[2]/2])

# Function to execute movement using linear interpolation
def execute_movement(start_pos, target_pos, num_steps=100, gripper_open=True, speed=1.0):
    """
    Execute robot movement using linear interpolation with adjustable speed.
    
    Args:
        start_pos: Starting position 
        target_pos: Target position
        num_steps: Total number of steps for the movement
        gripper_open: Whether gripper should be open (True) or closed (False)
        speed: Speed multiplier (1.0 is normal, >1.0 is faster, <1.0 is slower)
    """
    # Calculate IK for the target position
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=target_pos,
        quat=np.array([0, 1, 0, 0]),
    )
    
    # Set gripper position
    if gripper_open:
        qpos[-2:] = 0.04  # Open gripper
    else:
        qpos[-2:] = 0.0   # Close gripper
    
    # Get current position
    current_qpos = franka.get_qpos()
    delta_qpos = qpos - current_qpos
    
    # Adjust number of steps based on speed
    # Faster speed = fewer steps, slower speed = more steps
    adjusted_steps = int(num_steps / speed)
    adjusted_steps = max(10, adjusted_steps)  # Ensure a minimum number of steps
    
    print(f"Moving from {start_pos} to {target_pos} with speed {speed}")
    print(f"Steps: {adjusted_steps}")
    
    # Execute the linear interpolation
    for i in range(adjusted_steps):
        # Simple linear interpolation
        alpha = i / (adjusted_steps - 1) if adjusted_steps > 1 else 1.0
        
        # Calculate interpolated joint positions
        interpolated_qpos = current_qpos + alpha * delta_qpos
        
        # Control DOFs
        franka.control_dofs_position(interpolated_qpos[:-2], motors_dof)
        franka.control_dofs_position(interpolated_qpos[-2:], fingers_dof)
        scene.step()
    
    # Ensure robot reaches final position
    for i in range(30):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(qpos[-2:], fingers_dof)
        scene.step()
    
    return qpos

# Get block position
block_position = np.array(start_cube_pos)

########################## Execute Pick and Place ##########################
print("Starting pick and place sequence...")

cube_positions_list = []

# 1. Move to position above the cube
position_above_cube = block_position + np.array([0.0, 0.0, 0.3])
execute_movement(franka.get_qpos(), position_above_cube, num_steps=150, gripper_open=True)

# 2. Lower gripper to cube
execute_movement(position_above_cube, block_position + np.array([0.0, 0.0, 0.11]), num_steps=100, gripper_open=True)
print("!!!!!!!!!!!!!!!!!", target_position + target_offset, "!!!!!!!!!!!!!!!!!!!!!!")
cube_positions_list.append(torch.norm((target_position + target_offset) - cube.get_pos().to(dtype=torch.float64)))

# 3. Close gripper to grasp cube
print("Closing gripper")
current_pos = franka.get_qpos()
franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
for i in range(50):
    scene.step()

# 4. Lift cube up
position_lift = block_position + np.array([0.0, 0.0, 0.3])
execute_movement(block_position + np.array([0.0, 0.0, 0.1]), position_lift, num_steps=120, gripper_open=False)

cube_positions_list.append(torch.norm((target_position + target_offset) - cube.get_pos().to(dtype=torch.float64)))

# 5. Move above target
position_above_target = target_pos + np.array([0.0, 0.0, 0.3])
execute_movement(position_lift, position_above_target, num_steps=200, gripper_open=False)

# 6. Lower to target
execute_movement(position_above_target, target_pos + np.array([0.0, 0.0, 0.1]), num_steps=120, gripper_open=False)

cube_positions_list.append(torch.norm((target_position + target_offset) - cube.get_pos().to(dtype=torch.float64)))

# 7. Open gripper to release cube
print("Opening gripper")
franka.control_dofs_position(np.array([0.04, 0.04]), fingers_dof)
for i in range(50):
    scene.step()

cube_positions_list.append(torch.norm((target_position + target_offset) - cube.get_pos().to(dtype=torch.float64)))

# 8. Move back up
execute_movement(target_pos + np.array([0.0, 0.0, 0.3]), position_above_target, num_steps=100, gripper_open=True)

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", cube_positions_list, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# Keep the simulation running
print("Pick and place sequence completed. Press Ctrl+C to exit.")
try:
    while True:
        scene.step()
        time.sleep(0.01)
except KeyboardInterrupt:
    print("Exiting simulation")
