import numpy as np
from numpy.linalg import inv
import copy
from collections import OrderedDict


class CAMERA:
    # constructor
    def __init__(self, *parent):
        # initialize
        self.parent = parent[0]
        # pixel coordinates
        self.pixels = self.return_pixel_coordinates()
        # initialize
        self.initialize_intrinsics()

    # PROJECT MODEL ####################################################################################################

    def project_corners_through_model(self):
        # initialize
        for panel in ['panel_1', 'panel_2', 'panel_3']:
            xyz1 = np.zeros((self.parent.charuco.board_corners[panel].shape[0], 4))
            xyz1[:, :3], xyz1[:, -1] = self.parent.charuco.board_corners[panel], 1
            # project to world coordinates
            xyz = np.matmul(self.parent.world.aXs_angle, np.matmul(self.parent.world.Rt, xyz1.T)).T
            self.parent.charuco.projected_board_corners_3d[panel] = xyz.copy()
            # normalize to unit sphere
            xyz = xyz / np.sqrt((xyz ** 2).sum(axis=1)).reshape(-1, 1)
            # project to lens distorted co-ordinates
            if np.logical_and(self.parent.int_vars['re-projection_mode'].get() == 'Apply',
                              self.parent.int_vars['re-projection_type'].get() == 'Rectilinear'):
                zoom = self.parent.default_settings['re-projection']['zoom_rectilinear']
                # to camera coordinates
                xy1 = xyz / (xyz[:, 2] + 1e-12).reshape(-1, 1)
                xy1[:, :2] = xy1[:, :2] / zoom
                # project to pixel coordinates
                xy1 = np.matmul(xy1, self.K.T)
            else:
                # project to image coordinates
                xy1 = self.spherical_rays_to_pixel_coords(xyz)
            self.parent.charuco.projected_board_corners[panel] = xy1[:, :2]

    # FUNCTIONS ########################################################################################################

    def initialize_intrinsics(self):
        # camera matrix
        self.K = self.return_camera_matrix()
        # radial distortion LUT
        self.LUT = self.return_radial_distortion_LUT()
        # sensor tilt vector
        self.tilt = self.return_sensor_tilt_vector()
        # pixel vectors unit sphere
        self.pixel_rays = {'xyz': self.return_pixel_vectors()}
        # initialize projection rays
        self.no_tilt_projection_rays = self.init_projection_rays_no_tilt()
        self.no_tilt_projection_fa = np.arccos(self.no_tilt_projection_rays[:, 2])
        # get interpolation grids
        self.interpolation_grids = self.return_interpolation_grids()
        # max vertical fov for background image
        self.max_vert_fov = self.get_max_vertical_fov()

    def return_pixel_vectors(self):
        xy1 = np.array([self.pixels['x'], self.pixels['y'], np.ones((len(self.pixels['x']),))]).T
        # remove radial distortion + sensor tilt
        return self.pixel_coords_to_spherical_rays(xy1)

    def return_interpolation_grids(self):
        # rectilinear interpolation grid
        rectilinear = self.rectilinear_interpolation_grid()
        curvilinear = self.curvilinear_interpolation_grid()
        spherical = self.spherical_interpolation_grid()
        return {'Rectilinear': rectilinear, 'Curvilinear': curvilinear, 'Spherical': spherical}

    # FUNCTIONS ########################################################################################################

    def radial_distortion(self, xyz):
        # field angle
        field_angle = np.arccos(xyz[:, 2])
        azimuth_angle = np.arctan2(xyz[:, 1], xyz[:, 0])
        # field angle
        radius = np.interp(field_angle, self.LUT['theta'], self.LUT['r'])
        # pixel vector on the image plane
        xy1 = np.array([radius * np.cos(azimuth_angle),
                        radius * np.sin(azimuth_angle), np.ones(len(field_angle), )])
        return xy1.T

    def radial_undistortion(self, xy1):
        # xy-radius
        r = np.sqrt(xy1[:, 0] ** 2 + xy1[:, 1] ** 2)
        # xy-azimuthal angle
        cosine_azi = xy1[:, 0] / r
        sine_azi = xy1[:, 1] / r
        # field angle
        field_angle = np.interp(r, self.LUT['r'], self.LUT['theta'])
        # pixel vector on the unit sphere
        xyz = np.array([np.sin(field_angle) * cosine_azi, np.sin(field_angle) * sine_azi, np.cos(field_angle)])
        return xyz.T

    def sensor_tilt(self, xy1, method='project_rays'):
        if np.abs(self.tilt['nz']) == 1:
            return xy1
        nx = self.tilt['nx']
        ny = self.tilt['ny']
        nz = self.tilt['nz']
        # shorthand notation
        xy1_copy = xy1.copy()
        px = xy1[:, 0]
        py = xy1[:, 1]
        # world-to-sensor | [eqns 4a & 4b]
        if method == 'project_rays':
            xy1_copy[:, 0] = (((nx ** 2 + nz * (nz - 1.0)) * px + nx * ny * py) / (nx * px + ny * py + nz)) / (nz - 1.0)
            xy1_copy[:, 1] = (((ny ** 2 + nz * (nz - 1.0)) * py + nx * ny * px) / (nx * px + ny * py + nz)) / (nz - 1.0)
        # sensor-to-world | [eqns 7a & 7b]
        if method == 'unproject_rays':
            xy1_copy[:, 0] = (((nx ** 2 + nz - 1.0) * px + nx * ny * py) / (nx * px + ny * py + 1.0)) / (nz - 1.0)
            xy1_copy[:, 1] = (((ny ** 2 + nz - 1.0) * py + nx * ny * px) / (nx * px + ny * py + 1.0)) / (nz - 1.0)
        return xy1_copy

    def return_sensor_tilt_vector(self):
        theta = self.parent.int_vars['tilt_angle'].get() * np.pi / 180
        azimuth = self.parent.int_vars['tilt_azimuth'].get() * np.pi / 180
        return {'nx': np.sin(theta) * np.cos(azimuth), 'ny': np.sin(theta) * np.sin(azimuth), 'nz': -np.cos(theta)}

    def return_radial_distortion_LUT(self, npts=2 ** 12):
        theta = np.linspace(0, 0.9 * np.pi, npts)
        if self.parent.int_vars['lens_distortion'].get() == 'Equidistant':
            r = theta
        elif self.parent.int_vars['lens_distortion'].get() == 'Equisolid':
            r = 2 * np.sin(theta / 2)
        elif self.parent.int_vars['lens_distortion'].get() == 'Orthographic':
            r = np.clip(np.cumsum(np.abs(np.gradient(np.sin(theta)))), 0, 10)
            clip_index = np.where((theta - np.pi / 2) ** 2 == ((theta - np.pi / 2) ** 2).min())[0][0]
            r[clip_index:], theta[clip_index:] = r[clip_index], theta[clip_index]
        elif self.parent.int_vars['lens_distortion'].get() == 'Stereographic':
            r = 2 * np.tan(theta / 2)
        elif self.parent.int_vars['lens_distortion'].get() == 'Polynomial':
            lens_poly = self.parent.default_settings['polynomial_distortion']
            r = theta + lens_poly['k2'] * theta ** 2 + lens_poly['k3'] * theta ** 3 + lens_poly['k4'] * theta ** 4
        # check polynomial integrity
        lens_poly = self.parent.default_settings['polynomial_distortion']
        r_poly = theta + lens_poly['k2'] * theta ** 2 + lens_poly['k3'] * theta ** 3 + lens_poly['k4'] * theta ** 4
        if np.gradient(r_poly).min() < 0:
            print('Polynomial distortion is not monotonically increasing')
            raise SystemExit
        return {'r': r, 'theta': theta}

    def get_max_vertical_fov(self, diagonal=False, npts=2 ** 12):
        theta = np.linspace(0, 0.9 * np.pi, npts)
        if diagonal:
            distance = np.sqrt((self.parent.int_vars['height'].get() / 2) ** 2
                               + (self.parent.int_vars['width'].get() / 2) ** 2)
        else:
            distance = self.parent.int_vars['height'].get() / 2
        poly = self.parent.default_settings['polynomial_distortion']
        f_max, f_min = max(self.K[0, 0], self.K[1, 1]), min(self.K[0, 0], self.K[1, 1])
        A_max = f_max * np.array([theta, 2 * np.sin(theta / 2), 2 * np.tan(theta / 2),
                                  theta + poly['k2'] * theta ** 2 + poly['k3'] * theta ** 3 + poly['k4'] * theta ** 4])
        A_min = f_min * np.array([theta, 2 * np.sin(theta / 2), 2 * np.tan(theta / 2),
                                  theta + poly['k2'] * theta ** 2 + poly['k3'] * theta ** 3 + poly['k4'] * theta ** 4])
        A_max = np.abs(A_max - distance)
        A_min = np.abs(A_min - distance)
        index_max = [np.where(A_max[a, :] == A_max[a, :].min())[0][0] for a in range(4)]
        index_min = [np.where(A_min[a, :] == A_min[a, :].min())[0][0] for a in range(4)]
        return max(theta[np.array(index_min).max()], theta[np.array(index_max).max()])

    def return_camera_matrix(self):
        h, w = self.parent.int_vars['height'].get(), self.parent.int_vars['width'].get()
        fx = self.parent.int_vars['focal_length'].get()
        fy = fx * float(self.parent.int_vars['yx_aspect_ratio'].get())
        oX = w / 2 - 0.5 + float(self.parent.int_vars['dX'].get())
        oY = h / 2 - 0.5 + float(self.parent.int_vars['dY'].get())
        # camera matrix
        K = np.array([[fx, 0, oX],
                      [0, fy, oY],
                      [0, 0, 1]])
        return K

    # __INIT__ #########################################################################################################
    def return_pixel_coordinates(self):
        h, w = self.parent.int_vars['height'].get(), self.parent.int_vars['width'].get()
        x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
        grid_xy = self.parent.default_settings['re-projection']['grid_xy']
        return {'x': x.flatten(), 'y': y.flatten(), 'h': int(h), 'w': int(w), 'grid_xy': grid_xy}

    def rectilinear_interpolation_grid(self):
        zoom = self.parent.default_settings['re-projection']['zoom_rectilinear']
        xy1 = np.array([self.pixels['x'], self.pixels['y'], np.ones((len(self.pixels['x'].flatten()),))]).T
        # remove camera matrix
        xy1 = np.matmul(xy1, inv(self.K).T)
        # zoom coordinates
        if zoom != 1.0:
            xy1[:, 0], xy1[:, 1] = xy1[:, 0] * zoom, xy1[:, 1] * zoom
        # normalize to unit sphere
        xyz = xy1 / np.sqrt((xy1 ** 2).sum(axis=1)).reshape(-1, 1)
        # project to image coordinates
        xy1 = self.spherical_rays_to_pixel_coords(xyz)
        return xy1

    def curvilinear_interpolation_grid(self):
        zoom = self.parent.default_settings['re-projection']['zoom_cylindrical']
        h, w = self.pixels['h'], self.pixels['w']
        # remove radial distortion + return spherical rays
        xyz = copy.deepcopy(self.no_tilt_projection_rays).reshape(h, w, 3)
        fa = self.no_tilt_projection_fa.reshape(h, w)
        # continue
        h_top_btm = [xyz[0, int(w / 2), 1] * zoom, xyz[-1, int(w / 2), 1] * zoom]
        fa_left_right = [-fa[int(h / 2), 0], fa[int(h / 2), -1]]
        # pixel grid
        theta, h = np.meshgrid(np.linspace(fa_left_right[0], fa_left_right[1], w),
                               np.linspace(h_top_btm[0], h_top_btm[1], h))
        # to cartesian
        y, z, x = h, np.cos(theta), np.sin(theta)
        xyz = np.array([x.flatten(), y.flatten(), z.flatten()]).T
        # Normalize to the unit sphere
        xyz = xyz / np.sqrt((xyz ** 2).sum(axis=1)).reshape(-1, 1)
        # project to image coordinates
        xy1 = self.spherical_rays_to_pixel_coords(xyz)
        return xy1

    def spherical_interpolation_grid(self):
        h, w = self.pixels['h'], self.pixels['w']
        # remove radial distortion + return spherical rays
        xyz = copy.deepcopy(self.no_tilt_projection_rays).reshape(h, w, 3)
        fa = self.no_tilt_projection_fa.reshape(h, w)
        # continue
        azi_top_btm = [np.arcsin(xyz[0, int(w / 2), 1]), np.arcsin(xyz[-1, int(w / 2), 1])]
        fa_left_right = [-fa[int(h / 2), 0], fa[int(h / 2), -1]]
        # pixel grid
        theta, phi = np.meshgrid(np.linspace(fa_left_right[0], fa_left_right[1], w),
                                 np.linspace(azi_top_btm[0], azi_top_btm[1], h))
        # to cartesian
        x, y, z = np.sin(theta) * np.cos(phi), np.sin(phi), np.cos(theta) * np.cos(phi)
        xyz = np.array([x.flatten(), y.flatten(), z.flatten()]).T
        # project to image coordinates
        xy1 = self.spherical_rays_to_pixel_coords(xyz)
        return xy1

    def spherical_rays_to_pixel_coords(self, xyz):
        # apply radial distortion
        xy1 = self.radial_distortion(xyz)
        # apply sensor tilt
        xy1 = self.sensor_tilt(xy1)
        # return pixel coordinates
        return np.matmul(xy1, self.K.T)

    def pixel_coords_to_spherical_rays(self, xy1):
        # remove camera matrix
        xy1 = np.matmul(xy1, inv(self.K).T)
        # remove sensor tilt
        xy1 = self.sensor_tilt(xy1, method='unproject_rays')
        # remove radial distortion + return spherical rays
        return self.radial_undistortion(xy1)

    def remove_sensor_tilt_from_pixel_rays(self, xyz):
        # to camera coordinates
        xy1 = self.radial_distortion(xyz)
        # remove sensor tilt
        xy1 = self.sensor_tilt(xy1, method='unproject_rays')
        # remove radial distortion
        xyz = self.radial_undistortion(xy1)
        return xyz

    def init_projection_rays_no_tilt(self):
        xy1 = np.array([self.pixels['x'], self.pixels['y'], np.ones((len(self.pixels['x']),))]).T
        # remove camera matrix
        xy1 = np.matmul(xy1, inv(self.K).T)
        # remove radial distortion + return spherical rays
        xyz = self.radial_undistortion(xy1)
        return xyz

    def get_ground_truth_camera_parameters(self):
        # lens distortion
        M = np.array([self.LUT['theta'] ** 2, self.LUT['theta'] ** 3, self.LUT['theta'] ** 4]).T
        if np.logical_and(self.parent.int_vars['re-projection_mode'].get() == 'Apply',
                          self.parent.int_vars['re-projection_type'].get() == 'Rectilinear'):
            # max field angle is 90 degrees
            measure = (self.LUT['theta'] - 0.75 * np.pi / 2) ** 2
            idx = np.where(measure == measure.min())[0][0]
            r, M = np.tan(self.LUT['theta'][:idx]), M[:idx, :]
            kappa = np.matmul(inv(np.matmul(M.T, M)), np.matmul(M.T, r - self.LUT['theta'][:idx]))
            # verify
            r1 = np.matmul(M, kappa) + self.LUT['theta'][:idx]
        else:
            kappa = np.matmul(inv(np.matmul(M.T, M)), np.matmul(M.T, self.LUT['r'] - self.LUT['theta']))
        # number of panels in virtual image
        panels = ['panel_1', 'panel_2', 'panel_3'] if self.parent.charuco.params['target_3d'].get() else ['panel_1']
        # ground truth camera parameters
        self.project_corners_through_model()
        # loop through panels
        _ext_dict_ = {}
        for panel in panels:
            # # get rotations and translation
            # R1, R2, t = self.parent.world.Rt[:, :3], self.parent.world.aXs_angle, self.parent.world.Rt[:, -1]
            # # project board corners
            # xyz_charuco = np.matmul(R1, self.parent.charuco.board_corners[panel].T) + t[:, np.newaxis]
            # self.parent.charuco.projected_board_corners_3d[panel] = np.matmul(R2, xyz_charuco).T
            # get projected panel corners
            xyz_projected = self.parent.charuco.projected_board_corners_3d[panel]
            # get initial position of panel corners
            xyz_initial = self.parent.charuco.board_corners[panel]
            # find rotation and translation
            R, t = self.find_rotation_translation(xyz_initial, xyz_projected)
            rpy = self.parent.calibrate.rot_matrix_to_rpy(R)
            # verify result
            if np.logical_or(np.abs(np.matmul(R, xyz_initial.T).T + t[np.newaxis, :] - xyz_projected).sum() > 1e-6,
                             np.abs(self.parent.calibrate.rpy_to_rot_matrix(rpy[0], rpy[1], rpy[2]) - R).sum() > 1e-6):
                print('Conversion error RPY to Rotation Matrix' + '\n')
                raise SystemExit
            _ext_dict_['image00_' + panel.replace('_', '0') + '_roll'] = rpy[0]
            _ext_dict_['image00_' + panel.replace('_', '0') + '_pitch'] = rpy[1]
            _ext_dict_['image00_' + panel.replace('_', '0') + '_yaw'] = rpy[2]
            _ext_dict_['image00_' + panel.replace('_', '0') + '_tX'] = t[0]
            _ext_dict_['image00_' + panel.replace('_', '0') + '_tY'] = t[1]
            _ext_dict_['image00_' + panel.replace('_', '0') + '_tZ'] = t[2]
        # camera matrix zoom rectilinear
        K = copy.deepcopy(self.K)
        if np.logical_and(self.parent.int_vars['re-projection_mode'].get() == 'Apply',
                          self.parent.int_vars['re-projection_type'].get() == 'Rectilinear'):
            K[0, 0] = K[0, 0] / self.parent.default_settings['re-projection']['zoom_rectilinear']
            K[1, 1] = K[1, 1] / self.parent.default_settings['re-projection']['zoom_rectilinear']
        self.eta_ground_truth = self.fill_eta_vector(K, kappa, _ext_dict_)
        return copy.deepcopy(self.eta_ground_truth)

    def fill_eta_vector(self, K, kappa, _ext_dict_):
        h, w = self.parent.image['Virtual Camera'].height, self.parent.image['Virtual Camera'].width
        # Create an ordered dictionary
        eta = OrderedDict()
        # focal length
        eta['f'] = copy.deepcopy(K[0, 0])
        # focal length aspect ratio
        eta['lambda'] = K[1, 1] / K[0, 0]
        # optical center offset dX
        eta['dX'] = K[0, 2] - np.round((w / 2 - 0.5), 6)
        # optical center offset dY
        eta['dY'] = K[1, 2] - np.round((h / 2 - 0.5), 6)
        # lens distortion coefficients
        eta['kappa_2'] = copy.deepcopy(kappa[0])
        # lens distortion coefficients
        eta['kappa_3'] = copy.deepcopy(kappa[1])
        # lens distortion coefficients
        eta['kappa_4'] = copy.deepcopy(kappa[2])
        # extrinsics
        for name in list(_ext_dict_):
            # assign
            eta[name] = copy.deepcopy(_ext_dict_[name])
        return eta

    @staticmethod
    def kabsch_algorithm(P, Q):
        """
        Kabsch algorithm to find the optimal rotation matrix that aligns P to Q.
        P and Q are Nx3 matrices of points.
        """
        # Centroid of P and Q
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)

        # Center the points
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q

        # Covariance matrix
        H = np.dot(P_centered.T, Q_centered)

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)

        # Rotation matrix
        R = np.dot(Vt.T, U.T)

        # check translation
        t1 = (centroid_Q - centroid_P).reshape(-1, 1)
        # verify
        delta = (Q - (np.matmul(R, (P - centroid_P).T).T + centroid_P) - t1.T)

        # Special reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        return R, t1

    def find_rotation_translation(self, xyz_initial, xyz_projected):
        """
        Find the rotation matrix and translation vector to move xyz_initial to xyz_projected.
        """
        # Compute the rotation matrix using Kabsch algorithm
        R, t1 = self.kabsch_algorithm(xyz_initial, xyz_projected)

        # translation
        t = (xyz_projected - np.matmul(R, xyz_initial.T).T).mean(axis=0)

        # verify
        delta = np.abs(xyz_projected - np.matmul(R, xyz_initial.T).T - t).max()

        return R, t
