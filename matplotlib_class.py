import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import KMeans
from cv2 import resize, INTER_LINEAR
import yaml
import os
from matplotlib import cm


class MATPLOTLIB:
    def __init__(self, parent, camera_obj='Security cameras.obj', rot_obj_x=-90, rot_obj_y=0, rot_obj_z=0, r=50):
        # super().__init__(master=parent)
        self.parent = parent
        self.camera_obj, self.angle_x, self.angle_y, self.angle_z = camera_obj, rot_obj_x, rot_obj_y, rot_obj_z
        self.radius = r
        self.grid_settings = self.parent.default_settings['image_lens_grid']
        # initialize the figure
        self.azim, self.elev = -123, 47
        self.lens_grid_figure_open = None
        self.sphere = self.delaunay_triangulation()
        self.unit_sphere = self.delaunay_triangulation(r=1)
        self.initialize_figure()

    def initialize_intrinsics_figure(self):
        # initialize the radial distortion figure
        self.initialize_radial_distortion_figure()
        # initialize the radial distortion sphere
        self.initialize_radial_distortion_sphere()

    def update_intrinsics_figure(self):
        self.update_radial_distortion_figure()
        self.update_radial_distortion_sphere()

    def initialize_extrinsics_figure(self):
        # initialize the 3D World figure
        pass

    def update_extrinsics_figure(self):
        pass

    @staticmethod
    def corners_to_edges(corners):
        # extract
        TL, TR, BL, BR = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
        # edges
        top = np.array([TL, TR]).T
        bottom = np.array([BL, BR]).T
        left = np.array([TL, BL]).T
        right = np.array([TR, BR]).T
        # concatenate
        edges = np.concatenate((top, bottom, left, right), axis=1)
        # homogeneous
        edges_xyz1 = np.ones((4, edges.shape[1]))
        edges_xyz1[:3, :] = edges
        return edges_xyz1

    @staticmethod
    def downsample_image(image, h, w):
        return resize(image, (w, h), interpolation=INTER_LINEAR)

    def project_model(self, panel='panel_1'):
        h, w = self.parent.photon.object[panel]['image'].shape[0], self.parent.photon.object[panel]['image'].shape[1]
        s = min(w, h) / 50  # down sample factor
        xmin = self.parent.photon.object[panel]['corners'][0, :].min()
        xmax = self.parent.photon.object[panel]['corners'][0, :].max()
        ymin = self.parent.photon.object[panel]['corners'][1, :].min()
        ymax = self.parent.photon.object[panel]['corners'][1, :].max()
        # image borders
        edges_xyz1 = self.corners_to_edges(self.parent.photon.object[panel]['corners'])
        # project edges through model
        edges_xyz = np.matmul(self.parent.world.aXs_angle, np.matmul(self.parent.world.Rt, edges_xyz1))
        # create plane
        # create x,y
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, int(w / s) + 1), np.linspace(ymin, ymax, int(h / s) + 1))
        zz, hP, wP = np.zeros(xx.shape), xx.shape[0], xx.shape[1]
        XYZ1 = np.concatenate((xx.flatten(), yy.flatten(), zz.flatten(), np.ones(zz.flatten().shape))).reshape(4, -1)
        # project plane
        XYZ = np.matmul(self.parent.world.aXs_angle, np.matmul(self.parent.world.Rt, XYZ1))
        # down sample image data
        image_data = resize(self.parent.photon.object[panel]['image'][:, :, 1], (int(w / s), int(h / s)),
                            interpolation=INTER_LINEAR)
        # project corners
        return {"edges": edges_xyz, "image": image_data / image_data.max(),
                "plane": [XYZ[0, :].reshape(hP, wP), XYZ[1, :].reshape(hP, wP), XYZ[2, :].reshape(hP, wP)]}

    def sparse_lens_grid(self, r=None):
        h, w = self.parent.default_settings['sensor']['height'], self.parent.default_settings['sensor']['width']
        if r is None:
            r = self.radius
        # pixel rays
        xyz = self.parent.camera.pixel_rays['xyz']
        x, y, z = xyz[:, 0].reshape(h, w), xyz[:, 1].reshape(h, w), xyz[:, 2].reshape(h, w)
        # grid
        gridX = np.linspace(0, w - 1, self.grid_settings['grid_xy'][0]).astype('int')
        gridY = np.linspace(0, h - 1, self.grid_settings['grid_xy'][1]).astype('int')
        # rescale and sample
        grid_h = {'x': r * x[:, gridX], 'y': r * y[:, gridX], 'z': r * z[:, gridX]}
        grid_w = {'x': r * x[gridY, :], 'y': r * y[gridY, :], 'z': r * z[gridY, :]}
        return grid_h, grid_w

    def plot_lens_grid(self):
        # print('Plotting lens grid')
        grid_h, grid_w = self.sparse_lens_grid(r=1)
        fig = plt.figure(figsize=(10.8, 10.2), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        # Set the title of the plot
        ax.set_title(self.parent.int_vars['lens_distortion'].get() + ' Lens Grid')

        # Plot the unit sphere
        ax.plot_trisurf(self.unit_sphere["x"], self.unit_sphere["z"], self.unit_sphere["y"],
                        triangles=self.unit_sphere["tri"].triangles, cmap='copper', linewidths=0.2, alpha=0.35)

        # Plot the lens grid
        for a in range(grid_h['x'].shape[1]):
            ax.plot(grid_h['x'][:, a], grid_h['y'][:, a], grid_h['z'][:, a], 'green', linewidth=4, alpha=0.8)
        for a in range(grid_w['x'].shape[0]):
            ax.plot(grid_w['x'][a, :], grid_w['y'][a, :], grid_w['z'][a, :], 'green', linewidth=4, alpha=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Minimize white space
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.tight_layout()

        # Set initial view angles
        ax.view_init(elev=self.elev, azim=self.azim)

        # Connect the event to track view angle
        fig.canvas.mpl_connect('button_release_event', self.record_view_angle)

        # Connect events to track when the figure is opened and closed
        fig.canvas.mpl_connect('draw_event', self.on_figure_open)
        fig.canvas.mpl_connect('close_event', self.on_figure_close)

        # Save the figure and axes
        self.lens_grid_figure = {'fig': fig, 'ax': ax}

        # show plot
        plt.show()

    def record_view_angle(self, event):
        ax = event.inaxes
        if ax:
            self.azim = ax.azim
            self.elev = ax.elev
            # print(f"View angle recorded: azim={self.azim}, elev={self.elev}")

    def on_figure_open(self, event):
        # print('Figure opened')
        # figure is open
        self.lens_grid_figure_open = True

    def on_figure_close(self, event):
        # print('Figure closed')
        # figure is open
        self.lens_grid_figure_open = False

    def update_lens_grid_figure(self):
        # Clear the existing plot
        self.lens_grid_figure['ax'].cla()

        # Set the title of the plot
        self.lens_grid_figure['ax'].set_title(self.parent.int_vars['lens_distortion'].get() + ' Lens Grid')

        # Recompute the lens grid
        grid_h, grid_w = self.sparse_lens_grid(r=1)

        # Replot the unit sphere
        self.lens_grid_figure['ax'].plot_trisurf(self.unit_sphere["x"], self.unit_sphere["z"], self.unit_sphere["y"],
                                                 triangles=self.unit_sphere["tri"].triangles, cmap='copper',
                                                 linewidths=0.2, alpha=0.35)

        # Replot the lens grid
        for a in range(grid_h['x'].shape[1]):
            self.lens_grid_figure['ax'].plot(grid_h['x'][:, a], grid_h['y'][:, a], grid_h['z'][:, a], 'green',
                                             linewidth=4, alpha=0.8)
        for a in range(grid_w['x'].shape[0]):
            self.lens_grid_figure['ax'].plot(grid_w['x'][a, :], grid_w['y'][a, :], grid_w['z'][a, :], 'green',
                                             linewidth=4, alpha=0.8)

        # Set labels
        self.lens_grid_figure['ax'].set_xlabel('X')
        self.lens_grid_figure['ax'].set_ylabel('Y')
        self.lens_grid_figure['ax'].set_zlabel('Z')

        # Set the view angles
        self.lens_grid_figure['ax'].view_init(elev=self.elev, azim=self.azim)

        # Redraw the figure
        self.lens_grid_figure['fig'].canvas.draw()

    # initialize the figure
    def initialize_figure(self, ):
        grid_h, grid_w = self.sparse_lens_grid()
        # declare figure
        fig1 = Figure(figsize=(15.8, 10.2), dpi=100)
        # SUBPLOT (3,3,1): global phase
        ax1 = fig1.add_subplot(111, projection='3d')
        # figure title
        ax1.set_title('World - Camera - Object')
        # Plot the plane
        sphere = [ax1.plot_trisurf(self.sphere["x"], self.sphere["z"], self.sphere["y"],
                                   triangles=self.sphere["tri"].triangles, cmap='copper', linewidths=0.2, alpha=0.35)]
        H_grid = [ax1.plot(grid_h['x'][:, a], grid_h['z'][:, a], grid_h['y'][:, a],
                           'green', linewidth=2, alpha=0.85)
                  for a in range(grid_h['x'].shape[1])]
        W_grid = [ax1.plot(grid_w['x'][a, :], grid_w['z'][a, :], grid_w['y'][a, :],
                           'green', linewidth=2, alpha=0.85)
                  for a in range(grid_w['x'].shape[0])]
        grid = [H_grid, W_grid]
        # Label the axes
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        ax1.set_zlabel('Y')
        # initialize frame
        edgeT, = ax1.plot([], [], [], 'gray', linewidth=4)  # frame axis 0
        edgeB, = ax1.plot([], [], [], 'gray', linewidth=4)  # frame axis 1
        edgeL, = ax1.plot([], [], [], 'gray', linewidth=4)  # frame axis 2
        edgeR, = ax1.plot([], [], [], 'gray', linewidth=4)  # frame axis 3
        # initialize plane
        xx, yy = np.meshgrid(np.linspace(-10, 10, 11), np.linspace(-10, 10, 11))
        zz = np.ones(xx.shape)
        plane = [ax1.plot_surface(xx, yy, zz, alpha=0.65)]
        ax1.view_init(azim=160, elev=15)
        self.figure = {'fig': fig1, 'ax1': ax1, 'frame': [edgeT, edgeB, edgeL, edgeR],
                       'plane': plane, 'sphere': sphere, 'grid': grid}

    def return_plot_limits(self, z, s_xy=0.66, panel='panel_1'):
        xmax = np.abs(self.parent.photon.object[panel]['corners'][0, :]).max()
        ymax = np.abs(self.parent.photon.object[panel]['corners'][1, :]).max()
        diag = np.sqrt(xmax ** 2 + ymax ** 2)
        return {'x': [(-z - diag) * s_xy, (z + diag) * s_xy], 'y': [(-z - diag) * s_xy, (z + diag) * s_xy],
                'z': [-diag, z + diag]}

    def update_figure(self, elev=-110, azim=0, roll=0):
        # project the model using current variables
        _limits_ = self.return_plot_limits(float(self.parent.ext_vars['z'].get()))
        _dict_ = self.project_model()
        # exchange [y, z] coordinates
        _limits_['y'], _limits_['z'] = _limits_['z'], _limits_['y']
        _dict_['edges'][1, :], _dict_['edges'][2, :] = _dict_['edges'][2, :].copy(), _dict_['edges'][1, :].copy()
        _dict_['plane'][1], _dict_['plane'][2] = _dict_['plane'][2].copy(), _dict_['plane'][1].copy()
        # shorthand variables
        fig, ax1, frame, plane = self.figure['fig'], self.figure['ax1'], self.figure['frame'], self.figure['plane']
        edgeT, edgeB, edgeL, edgeR = frame[0], frame[1], frame[2], frame[3]
        # continue
        top, bottom = _dict_['edges'][:, 0:2], _dict_['edges'][:, 2:4]
        left, right = _dict_['edges'][:, 4:6], _dict_['edges'][:, 6:8]
        # (x, y) coordinates
        edgeT.set_data(top[0, :], top[1, :])
        edgeB.set_data(bottom[0, :], bottom[1, :])
        edgeL.set_data(left[0, :], left[1, :])
        edgeR.set_data(right[0, :], right[1, :])
        # z coordinates
        edgeT.set_3d_properties(top[2, :])
        edgeB.set_3d_properties(bottom[2, :])
        edgeL.set_3d_properties(left[2, :])
        edgeR.set_3d_properties(right[2, :])
        # alpha
        self.figure['plane'][0].remove()
        self.figure['plane'] = [ax1.plot_surface(_dict_['plane'][0], _dict_['plane'][1], _dict_['plane'][2], alpha=0.65,
                                                 rstride=1, cstride=1, facecolors=plt.cm.BrBG(_dict_['image']))]
        # update the sphere
        self.figure['sphere'][0].remove()
        self.figure['sphere'] = [ax1.plot_trisurf(self.sphere["x"], self.sphere["z"], self.sphere["y"],
                                                  triangles=self.sphere["tri"].triangles,
                                                  cmap='copper', linewidths=0.2, alpha=0.35)]
        grid_h, grid_w = self.sparse_lens_grid()
        [line[0].remove() for line in self.figure['grid'][0]]
        [line[0].remove() for line in self.figure['grid'][1]]
        H_grid = [ax1.plot(grid_h['x'][:, a], grid_h['z'][:, a], grid_h['y'][:, a],
                           'green', linewidth=2, alpha=0.85)
                  for a in range(grid_h['x'].shape[1])]
        W_grid = [ax1.plot(grid_w['x'][a, :], grid_w['z'][a, :], grid_w['y'][a, :],
                           'green', linewidth=2, alpha=0.85)
                  for a in range(grid_w['x'].shape[0])]
        self.figure['grid'] = [H_grid, W_grid]
        # limits
        ax1.set_xlim(-250, 250)
        ax1.set_ylim(-75, 275)
        ax1.set_zlim(-250, 255)
        # axis equal
        ax1.set_aspect('equal')
        ax1.invert_zaxis()

    # initialize the unit sphere axes
    def init_unit_sphere_axes(self, ax1s):
        # surface
        ax1s.plot_trisurf(self.sphere["x"], self.sphere["y"], self.sphere["z"],
                          triangles=self.sphere["tri"].triangles,
                          cmap='copper', linewidths=0.2, alpha=0.7)
        # axis ticks # axis labels
        ax1s.set_xlabel("x")
        ax1s.set_ylabel("y")
        ax1s.set_zlabel("z")
        # ax1s.set_aspect('equal')

    def initialize_camera_object(self):
        file_path = os.path.join('obj', self.camera_obj)
        vertices, faces = self.load_obj(file_path)
        vertices = self.normalize_vertices(vertices, scale=100)

        # Downsample the vertices
        vertices, faces = self.downsample_vertices(vertices, faces, n_clusters=303)

        # Rotate the vertices
        vertices = self.rotate_vertices(vertices, angle_x=self.angle_x, angle_y=self.angle_y, angle_z=self.angle_z)

        return vertices, faces

    # delunay triangulation
    def delaunay_triangulation(self, r=None, vertices=44):
        if r is None:
            r = self.radius
        u, v = np.meshgrid(np.linspace(0, np.pi, vertices).astype('float64'),
                           np.linspace(0, 2 * np.pi, vertices).astype('float64'))
        # surface
        x = np.ravel(r * np.sin(u) * np.cos(v))
        y = np.ravel(r * np.sin(u) * np.sin(v))
        z = np.ravel(r * np.cos(u))
        # delunay triangulation
        return {"x": x, "y": y, "z": z, "tri": Triangulation(np.ravel(u), np.ravel(v))}

    @staticmethod
    def delaunay_triangulate_vertices(vertices):
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        u, v = np.arctan2(np.sqrt(x ** 2 + y ** 2), z), np.arctan2(y, x)
        return {"x": x, "y": y, "z": z, "tri": Triangulation(np.ravel(u), np.ravel(v))}

    @staticmethod
    def load_obj(file_path):
        vertices = []
        faces = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    vertices.append(list(map(float, line.strip().split()[1:4])))
                elif line.startswith('f '):
                    faces.append([int(idx.split('/')[0]) - 1 for idx in line.strip().split()[1:4]])
        return np.array(vertices), np.array(faces)

    @staticmethod
    def normalize_vertices(vertices, scale=1.0):
        vertices = (vertices - (vertices.max(0) + vertices.min(0)) / 2) / max(vertices.max(0) - vertices.min(0))
        return vertices * scale

    @staticmethod
    def rotate_vertices(vertices, angle_x=90, angle_y=90, angle_z=90):
        def rotation_matrix_x(angle):
            rad = np.radians(angle)
            return np.array([[1, 0, 0],
                             [0, np.cos(rad), -np.sin(rad)],
                             [0, np.sin(rad), np.cos(rad)]])

        def rotation_matrix_y(angle):
            rad = np.radians(angle)
            return np.array([[np.cos(rad), 0, np.sin(rad)],
                             [0, 1, 0],
                             [-np.sin(rad), 0, np.cos(rad)]])

        def rotation_matrix_z(angle):
            rad = np.radians(angle)
            return np.array([[np.cos(rad), -np.sin(rad), 0],
                             [np.sin(rad), np.cos(rad), 0],
                             [0, 0, 1]])

        R_x = rotation_matrix_x(angle_x)
        R_y = rotation_matrix_y(angle_y)
        R_z = rotation_matrix_z(angle_z)

        vertices = vertices @ R_x.T
        vertices = vertices @ R_y.T
        vertices = vertices @ R_z.T

        return vertices

    @staticmethod
    def downsample_vertices(vertices, faces, n_clusters=1000):
        # Apply k-means clustering to downsample vertices
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(vertices)
        downsampled_vertices = kmeans.cluster_centers_

        # Map original vertices to the nearest cluster center
        labels = kmeans.labels_
        vertex_map = {i: labels[i] for i in range(len(vertices))}

        # Adjust faces to map to the new downsampled vertices
        downsampled_faces = []
        for face in faces:
            try:
                downsampled_faces.append([vertex_map[idx] for idx in face])
            except KeyError:
                continue

        return np.array(downsampled_vertices), np.array(downsampled_faces)
