import numpy as np
from cv2 import polylines, imread, line
import glob
import os
from random import choice
from modules.gpu_utils import xp, to_gpu, to_cpu


class PHOTON:
    # constructor
    def __init__(self, *parent):
        self.counter = 0
        self.parent = parent[0]
        self.object = {'panel_1': None}
        self.world_background = imread(choice(glob.glob(os.path.join('background', '*'))))[:, :, ::-1]
        self.initialize_background()

    # FUNCTIONS ########################################################################################################

    def initialize_background(self):
        xyz = self.parent.camera.pixel_rays['xyz_gpu']
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        phi, height_y = xp.arctan2(x, z), y / xp.sqrt(x ** 2 + z ** 2)
        phi_abs_max = float(xp.abs(phi).max())
        # image coordinates
        max_vert_fov = self.parent.camera.max_vert_fov
        h, w = self.world_background.shape[0], self.world_background.shape[1]
        h2_radians = 1 / np.cos(min(1.025 * max_vert_fov, np.pi / 2 * 0.99))
        w2_radians = h2_radians / h * w
        # check if background image is too small
        if w2_radians < phi_abs_max:
            # rescale
            w2_radians_new = w2_radians * (1.01 * phi_abs_max / w2_radians)
            h2_radians_new = h2_radians * (w2_radians_new / w2_radians)
            h2_radians, w2_radians = h2_radians_new, w2_radians_new
        # continue
        yI, xI = xp.linspace(-h2_radians, h2_radians, h), xp.linspace(-w2_radians, w2_radians, w)
        # interp points
        pts_y = xp.ones(len(height_y)) * (-1)
        pts_y_mask = xp.logical_and(height_y >= yI.min(), height_y <= yI.max())
        try:
            pts_x = xp.interp(phi, xI, xp.linspace(0, w - 1, w))
            pts_y[pts_y_mask] = xp.interp(height_y[pts_y_mask], yI, xp.linspace(0, h - 1, h))
        except:
            print('Error: Background image is too small.')
        # biLinear interpolate the background image; stays GPU-resident as the base layer reused every frame
        self.sensor_rgb_out = self.bilinear_interp_image(to_gpu(self.world_background), xp.array([pts_x, pts_y]))

    def to_rgb(self):
        self.counter = self.counter + 1
        print('image generation %d' % self.counter)
        # 3D target
        target_3d = np.logical_and(self.parent.charuco.params['target_3d'].get(),
                                   self.parent.charuco.params['ChArUco'].get())
        panels = ['panel_1', 'panel_2', 'panel_3'] if target_3d else ['panel_1']
        # Sensor
        h, w = self.parent.default_settings['sensor']['height'], self.parent.default_settings['sensor']['width']
        sensor_rgb_out = xp.copy(self.sensor_rgb_out).astype('uint8')
        scale = self.object['panel_1']['pixel2world_scale']
        for panel in panels:
            sensor_indices = xp.arange(h * w)
            # Real World Image corner (X/Y, max/min) coordinates
            X_min = self.object[panel]['corners'][0, :].min()
            X_max = self.object[panel]['corners'][0, :].max()
            Y_min = self.object[panel]['corners'][1, :].min()
            Y_max = self.object[panel]['corners'][1, :].max()
            xyz = self.parent.camera.pixel_rays['xyz_gpu'].T
            R1, R2 = to_gpu(self.parent.world.Rt[:, :3]), to_gpu(self.parent.world.aXs_angle)
            t = to_gpu(self.parent.world.Rt[:, -1])
            norm, origin = to_gpu(self.object[panel]['normal']), to_gpu(self.object[panel]['origin'])  # normal and origin
            # 1)
            n, p0 = xp.matmul(R2, xp.matmul(R1, norm)), xp.matmul(R2, xp.matmul(to_gpu(self.parent.world.Rt), origin))
            # 2) in mask
            # project masks
            rays_mask = self.estimate_image_masks(xyz, n, panel=panel)
            xyz_masked = xyz[:, rays_mask]
            # 3) scale factor
            d = (p0 * n).sum() / (xyz_masked * n).sum(axis=0)
            # 4a) back project camera rays
            points_world_origin = xp.matmul(R1.T, xp.matmul(R2.T, xyz_masked * d) - t.reshape(-1, 1))
            # 4b) 'correct rotation' for side and bottom panels:
            if panel == 'panel_2':
                X_new = -points_world_origin[2, :] - scale * self.object[panel]['x0']
                points_world_origin[0, :], points_world_origin[2, :] = X_new, 0
            elif panel == 'panel_3':
                Y_new = -points_world_origin[2, :] - scale * self.object[panel]['y0']
                points_world_origin[1, :], points_world_origin[2, :] = Y_new, 0
            # 5) second mask
            in_mask_X = xp.logical_and(points_world_origin[0, :] > X_min, points_world_origin[0, :] < X_max)
            in_mask_Y = xp.logical_and(points_world_origin[1, :] > Y_min, points_world_origin[1, :] < Y_max)
            in_mask_XY = xp.logical_and(in_mask_X, in_mask_Y)
            # 6) camera rays to pixel coordinates
            pixels_xy = points_world_origin[:, in_mask_XY] / self.object[panel]['pixel2world_scale']
            # pixels_xy[1, :] = pixels_xy[1, ::-1] # reverse y-direction
            pixels_xy[0, :] = pixels_xy[0, :] + self.object[panel]['x0']
            pixels_xy[1, :] = pixels_xy[1, :] + self.object[panel]['y0']
            # 7) sensor indices
            sensor_indices = sensor_indices[rays_mask][in_mask_XY]
            # 8) biLinear interpolate the target image
            rgb_interp = self.bilinear_interp_image(self.object[panel]['image_gpu'], pixels_xy)
            # 9) add to sensor image
            sensor_rgb_out[sensor_indices, :] = rgb_interp

        # 10) reshape output
        sensor_rgb_out = sensor_rgb_out.reshape(h, w, 3)
        # 11) project lens grid
        if self.parent.int_vars['re-projection_type'].get() != 'None':
            sensor_rgb_out = self.project_lens_grid(sensor_rgb_out)
        if self.parent.int_vars['lens_distortion'].get() == 'Orthographic':
            # mask reflections
            if self.parent.int_vars['re-projection_mode'].get()!='Apply':
                reflections_mask = xyz[2, :] < 1e-3
                noise = (xp.random.rand(int(reflections_mask.sum()), 3) * 88).astype('uint8')
                sensor_rgb_out.reshape(h * w, 3)[reflections_mask] = noise
        if self.parent.plotting.lens_grid_figure_open:
            sensor_rgb_out = self.lens_grid_to_image(to_cpu(sensor_rgb_out), h, w)

        return to_cpu(sensor_rgb_out.reshape(h, w, 3))

    def lens_grid_to_image(self, sensor_rgb_out, h, w):
        sensor_rgb_out = sensor_rgb_out.reshape(h, w, 3)
        # grid
        gridX = np.linspace(0, w - 1, self.parent.default_settings['image_lens_grid']['grid_xy'][0]).astype('int')
        gridY = np.linspace(0, h - 1, self.parent.default_settings['image_lens_grid']['grid_xy'][1]).astype('int')
        # write interpolation grids to images
        for x0 in gridX:
            sensor_rgb_out = line(sensor_rgb_out, (x0, 0), (x0, h - 1), (99, 255, 99), thickness=2)
        for y0 in gridY:
            sensor_rgb_out = line(sensor_rgb_out, (0, y0), (w - 1, y0), (99, 255, 99), thickness=2)
        return sensor_rgb_out.reshape(h * w, 3)

    def project_lens_grid(self, image_rgb):
        re_type = self.parent.int_vars['re-projection_type'].get()
        if self.parent.int_vars['re-projection_mode'].get() == 'Show':
            grid = to_cpu(self.parent.camera.interpolation_grids_gpu[re_type])
            image_rgb = self.write_grid_to_image(to_cpu(image_rgb), grid.T)
        elif self.parent.int_vars['re-projection_mode'].get() == 'Apply':
            grid = self.parent.camera.interpolation_grids_gpu[re_type]
            image_rgb = self.bilinear_interp_image(image_rgb, grid.T)
            image_rgb = image_rgb.reshape(self.parent.camera.pixels['h'], self.parent.camera.pixels['w'], 3)
        return image_rgb.astype('uint8')

    def write_grid_to_image(self, image_rgb, grid, rgb=(237, 247, 105)):
        grid_xy = self.parent.camera.pixels['grid_xy']
        h, w = self.parent.camera.pixels['h'], self.parent.camera.pixels['w']
        # continue
        X_grid = grid[0, :].reshape(self.parent.camera.pixels['h'], self.parent.camera.pixels['w'])
        Y_grid = grid[1, :].reshape(self.parent.camera.pixels['h'], self.parent.camera.pixels['w'])
        # Find indices list
        iW_list = (np.linspace(0, w - 1, grid_xy[0] + 1) + 0.5).astype('int')
        iH_list = (np.linspace(0, h - 1, grid_xy[1] + 1) + 0.5).astype('int')
        # write interpolation grids to images
        for iW in iW_list:
            pts = np.array([X_grid[:, iW].flatten(), Y_grid[:, iW].flatten()]).T.reshape(-1, 1, 2) + 0.5
            image_rgb = polylines(image_rgb, np.int32([pts]), False, rgb, 2)
        for iH in iH_list:
            pts = np.array([X_grid[iH, :].flatten(), Y_grid[iH, :].flatten()]).T.reshape(-1, 1, 2) + 0.5
            image_rgb = polylines(image_rgb, np.int32([pts]), False, rgb, 2)
        return image_rgb

    @staticmethod
    def bilinear_interp_image(image_rgb, pixels_xy):
        h, w = image_rgb.shape[0], image_rgb.shape[1]
        image_rgb = image_rgb.reshape(h * w, 3).astype('int')
        # interp points
        int_x, int_y = pixels_xy[0, :].astype('int'), pixels_xy[1, :].astype('int')
        dx, dy = (pixels_xy[0, :] - int_x).reshape(-1, 1), (pixels_xy[1, :] - int_y).reshape(-1, 1)
        # unit square
        xy00 = int_x + int_y * w
        xy10 = int_x + 1 + int_y * w
        xy01 = int_x + (int_y + 1) * w
        xy11 = int_x + 1 + (int_y + 1) * w
        # mask indices outside frame
        mask_in = xp.logical_and(xy00 + w + 1 < h * w, xy00 > 0)
        # unit square rgb values
        f00, f10 = image_rgb[xy00[mask_in], :], image_rgb[xy10[mask_in], :]
        f01, f11 = image_rgb[xy01[mask_in], :], image_rgb[xy11[mask_in], :]
        # interp rgb values
        values = xp.zeros((pixels_xy.shape[1], 3), dtype='int')
        dX, dY = dx[mask_in], dy[mask_in]
        values[mask_in, :] = (f00 + (f10 - f00) * dX + (f01 - f00) * dY + (f11 - f10 - f01 + f00) * dX * dY + 0.5)
        return xp.clip(values, 0, 255).astype('int')

    def estimate_image_masks(self, xyz_rays, n, panel='panel_1'):
        xyz = xp.concatenate((to_gpu(self.object[panel]['normal']),
                              to_gpu(self.object[panel]['origin'][:3, :]),
                              to_gpu(self.object[panel]['corners'])), axis=1)
        xyz1 = xp.ones((4, xyz.shape[1]))
        xyz1[:3, :] = xyz
        # project into world
        XYZ = xp.matmul(to_gpu(self.parent.world.aXs_angle), xp.matmul(to_gpu(self.parent.world.Rt), xyz1))
        # normalize
        xyz_norm = XYZ / xp.sqrt((XYZ ** 2).sum(axis=0))
        # field angles
        fa = xp.arccos(xyz_norm[2, 2:])
        # else:
        mask = xp.arccos(self.parent.camera.pixel_rays['xyz_gpu'][:, 2]) <= fa.max()
        # assign
        in_mask = ((xyz_rays * n).sum(axis=0) < 0)
        if panel == 'panel_1':
            return xp.logical_and(mask, in_mask)
        else:
            return in_mask  # np.ones(len(in_mask), dtype=bool)

    # INITIALIZATION ###################################################################################################

    def load_object_image(self, image_rgb, charuco=False, panel='panel_1'):
        if np.logical_and(charuco is True, panel != 'panel_1'):
            image_rgb = np.repeat(image_rgb.reshape(image_rgb.shape[0], image_rgb.shape[1], 1), 3, axis=2)
        if charuco is False:
            pixel2world_scale = self.parent.default_settings['real_world_image_width_mm'] / image_rgb.shape[1]
        else:
            sq_x = self.parent.charuco.params['squares_x'].get()
            sq_length_mm = self.parent.charuco.params['square_length_mm'].get()
            pixel2world_scale = sq_length_mm * sq_x / image_rgb.shape[1]
        # object properties
        self.object[panel] = self.fill_object(image_rgb, pixel2world_scale, panel=panel)

    def fill_object(self, image_rgb, scale, panel='panel_1'):
        h, w = image_rgb.shape[0], image_rgb.shape[1]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        y0, x0 = h / 2 - 0.5, w / 2 - 0.5
        X = (x - x0) * scale
        Y = (y - y0)[::-1, :] * scale  # reverse y-dir
        Z = 0 * np.ones(x.shape)
        if panel == 'panel_1':
            # normal and origin
            normal = np.array([0, 0, -1]).reshape(-1, 1)
            origin = np.array([0, 0, 0, 1]).reshape(-1, 1)
            # real world coords
            self.panel_1_height_center_mm = y0 * scale
            self.panel_1_width_center_mm = x0 * scale
            # X = (x - x0) * scale
            # Y = (y - y0)[::-1, :] * scale  # reverse y-dir
            # Z = 0 * np.ones(x.shape)
        elif panel == 'panel_2':
            # normal and origin
            normal = np.array([-1, 0, 0]).reshape(-1, 1)
            origin = np.array([self.panel_1_width_center_mm, 0, -y0 * scale, 1]).reshape(-1, 1)
            # real world coords
            # X = np.ones(x.shape) * x0 * scale
            # Y = (y - y0)[::-1, :] * scale  # reverse y-dir
            # Z = - x * scale
        elif panel == 'panel_3':
            # normal and origin
            normal = np.array([0, -1, 0]).reshape(-1, 1)
            origin = np.array([0, self.panel_1_height_center_mm, -x0 * scale, 1]).reshape(-1, 1)
            # real world coords
            # X = (x - x0) * scale
            # Y = np.ones(x.shape) * y0 * scale
            # Z = - y * scale
        # fill object dictionary
        obj = {'image': image_rgb, 'image_gpu': to_gpu(image_rgb), 'pixel2world_scale': scale, 'h': h, 'w': w, 'y0': y0, 'x0': x0,
               'pixel_coords': {'x': x, 'y': y},  # 'world_coords': {'x': X, 'y': Y, 'z': Z},
               'normal': normal, 'origin': origin,
               'corners': np.array([[X[0, 0], Y[0, 0], Z[0, 0]], [X[0, -1], Y[0, -1], Z[0, -1]],
                                    [X[-1, 0], Y[-1, 0], Z[-1, 0]], [X[-1, -1], Y[-1, -1], Z[-1, -1]]]).T}
        # interpolate image
        return obj
