import jax.numpy as jnp

STATE_FIELDS = {
    'Pendulum-v1': ('theta', 'theta_dot'),
    'MountainCarContinuous-v0': ('position', 'velocity'),
    'Reacher-misc': ('angles', 'angle_vels'),
    'CartPoleContinuous-v0': ('x', 'x_dot', 'theta', 'theta_dot'),
    'CartPoleSwingUp-v0': ('x', 'x_dot', 'theta', 'theta_dot'),
    'PointRobot-misc': ('pos',),
    'Swimmer-misc': ('xy', 'xy_vel'),
    'AcrobotContinuous-v0': ('joint_angle1', 'joint_angle2', 'velocity_1', 'velocity_2'),
    'PlanarQuadrotor-misc': ('x', 'z', 'theta', 'x_dot', 'z_dot', 'theta_dot'),
}

ANGLE_FIELDS = {
    'Pendulum-v1': ('theta',),
    'Reacher-misc': ('angles',),
    'CartPoleContinuous-v0': ('theta',),
    'CartPoleSwingUp-v0': ('theta',),
    'AcrobotContinuous-v0': ('joint_angle1', 'joint_angle2'),
    'PlanarQuadrotor-misc': ('theta',),
}

STATE_BOUNDS = {
    'Pendulum-v1': lambda p: ((-jnp.pi, -p.max_speed), (jnp.pi, p.max_speed)),
    'MountainCarContinuous-v0': lambda p: ((p.min_position, -p.max_speed), (p.max_position, p.max_speed)),
}
