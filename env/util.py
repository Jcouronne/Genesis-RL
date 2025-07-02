import numpy as np

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = np.empty(4)
    q[0] = cr * cp * cy + sr * sp * sy  # w
    q[1] = sr * cp * cy - cr * sp * sy  # x
    q[2] = cr * sp * cy + sr * cp * sy  # y
    q[3] = cr * cp * sy - sr * sp * cy  # z
    return q
