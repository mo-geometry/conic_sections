import numpy as np


class WORLD:
    # constructor
    def __init__(self, *parent):
        self.ext_vars = parent[0].ext_vars
        # initialize extrinsics
        self.initialize_extrinsics()

    # FUNCTIONS ########################################################################################################

    def initialize_extrinsics(self, ):
        # generate rotation matrices
        self.Rt = self.generate_Rt()
        # generate axis-angle
        self.aXs_angle = self.generate_axis_angle()

    # ROTATION MATRICES ################################################################################################

    def generate_axis_angle(self):
        fa = self.ext_vars['field_angle'].get() * np.pi / 180
        azi = self.ext_vars['azimuth'].get() * np.pi / 180
        rot = self.ext_vars['rotation'].get() * np.pi / 180
        nx = np.cos(rot) * np.cos(azi)
        ny = np.cos(rot) * np.sin(azi)
        nz = np.sin(rot)
        # quaternion
        a, b, c, d = np.cos(fa / 2), nx * np.sin(fa / 2), ny * np.sin(fa / 2), nz * np.sin(fa / 2)
        return self.quaternion_to_rotation_matrix([a, b, c, d])

    def generate_Rt(self):
        # translation
        t = np.array([float(self.ext_vars['x'].get()),
                      float(self.ext_vars['y'].get()),
                      float(self.ext_vars['z'].get())])
        # rotation matrices
        roll = self.rotor_z(self.ext_vars['roll'].get() * np.pi / 180)
        pitch = self.rotor_x(self.ext_vars['pitch'].get() * np.pi / 180)
        yaw = self.rotor_y(self.ext_vars['yaw'].get() * np.pi / 180)
        # Euler convention
        if self.ext_vars['rpy_convention'].get() == 'yaw-pitch-roll':
            R = np.matmul(np.matmul(roll, pitch), yaw)
        elif self.ext_vars['rpy_convention'].get() == 'pitch-yaw-roll':
            R = np.matmul(np.matmul(roll, yaw), pitch)
        elif self.ext_vars['rpy_convention'].get() == 'roll-yaw-pitch':
            R = np.matmul(np.matmul(pitch, yaw), roll)
        elif self.ext_vars['rpy_convention'].get() == 'yaw-roll-pitch':
            R = np.matmul(np.matmul(pitch, roll), yaw)
        elif self.ext_vars['rpy_convention'].get() == 'pitch-roll-yaw':
            R = np.matmul(np.matmul(yaw, roll), pitch)
        elif self.ext_vars['rpy_convention'].get() == 'roll-pitch-yaw':
            R = np.matmul(np.matmul(yaw, pitch), roll)
        return np.concatenate((R, t[:, np.newaxis]), axis=1)

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        a, b, c, d = q[0], q[1], q[2], q[3]
        R = np.zeros((3, 3))
        # rotation matrix
        R[0, 0] = a ** 2 + b ** 2 - c ** 2 - d ** 2
        R[1, 1] = a ** 2 - b ** 2 + c ** 2 - d ** 2
        R[2, 2] = a ** 2 - b ** 2 - c ** 2 + d ** 2
        R[0, 1] = 2 * (b * c - a * d)
        R[1, 0] = 2 * (b * c + a * d)
        R[0, 2] = 2 * (b * d + a * c)
        R[2, 0] = 2 * (b * d - a * c)
        R[1, 2] = 2 * (c * d - a * b)
        R[2, 1] = 2 * (c * d + a * b)
        return R

    @staticmethod
    def rotor_x(angle):  # pitch
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

    @staticmethod
    def rotor_y(angle):  # yaw
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])

    @staticmethod
    def rotor_z(angle):  # roll
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
