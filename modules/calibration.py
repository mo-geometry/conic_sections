import numpy as np
from scipy.linalg import inv
from collections import OrderedDict
import sys
import copy
from cv2 import circle, imwrite, putText, FONT_HERSHEY_COMPLEX, getTextSize, rectangle
from cv2 import resize, VideoWriter, VideoWriter_fourcc
import os
from shutil import rmtree
import tkinter.messagebox as messagebox
import tkinter as tk
from tkinter import Toplevel, Text, ttk


class CALIBRATE:
    # constructor
    def __init__(self, *parent):
        self.parent = parent[0]
        self.camera_model = {}
        self.xy1_projected = {}
        self.stage_ctr, self.calibration_stage_images = 0, []
        self.eta_ground_truth, self.eta = None, None

    # FUNCTIONS ########################################################################################################

    def initialization(self):
        # initialize camera matrix and extrinsics
        K, R, t = self.initialize_KRt()
        # initialize distortion coefficients
        kappa = self.initialize_lens_distortion(K, R, t)
        # change rotor to roll-pitch-yaw
        extrinsics = self.extrinsics_roll_pitch_yaw_from_rotor_translation(R, t)
        # initialize optimization variables
        self.eta = self.initialize_optimization_variables(K, kappa, extrinsics)
        # update camera model
        self.update_camera_model()
        # project camera model
        self.project_camera_model()
        # update the Jacobian
        self.update_jacobian()
        # update objective function
        self.update_objective_function()
        # print stage image
        self.print_stage_images(reset=True)

    @staticmethod
    def write_reprojected_to_image(image, all_corners):
        if len(all_corners) > 0:
            for xy in all_corners:
                x, y = int(xy[0] + 0.5), int(xy[1] + 0.5)
                image = circle(image, (x, y), radius=2, color=(88, 88, 220), thickness=-1)
                image = circle(image, (x, y), radius=11, color=(101, 255, 101), thickness=3)
        return image

    @staticmethod
    def save_stage_image(image, file_name):
        # save image
        imwrite(f"stage_images/{file_name}", image)

    @staticmethod
    def create_directory(directory, delete_dir=False):
        # check if directory exists
        if os.path.exists(directory):
            if delete_dir:
                rmtree(directory)
            os.makedirs(directory)
        elif not os.path.exists(directory):
            os.makedirs(directory)

    def print_stage_images(self, downsample=True, reset=False):
        if np.logical_and(not self.parent.default_settings['stage_images'],
                          not self.parent.default_settings['stage_images_to_video']):
            return
        if reset:
            self.stage_ctr = 0
            self.calibration_stage_images = []
            self.create_directory('stage_images', delete_dir=True)
        self.stage_ctr = self.stage_ctr + 1
        first_image = False
        for image_name, image_tag in zip(self.calibration_images, self.xy1_projected):
            if first_image:
                break
            image = copy.deepcopy(self.calibration_images[image_name])
            points_xy = np.concatenate([self.xy1_projected[image_tag][p] for p in list(self.xy1_projected[image_tag])])
            # write projected points
            image = self.write_reprojected_to_image(image, points_xy)
            # write text
            tag = '%.9f' % self.objective_function if self.use_ground_truth_points else '%.1f' % self.objective_function
            text = 'Total squared error = ' + tag
            font = FONT_HERSHEY_COMPLEX
            font_scale = 1
            thickness = 4
            color_box = (200, 200, 200)  # Light gray color
            color_text = (66, 66, 66)  # Black color for text
            padding = 5

            # Get text size
            (text_width, text_height), baseline = getTextSize(text, font, font_scale, thickness)
            x, y = 12, 62  # Text position

            # Draw filled rectangle with padding
            top_left = (x - padding, y - text_height - padding - 5)
            bottom_right = (x + text_width + padding, y + baseline + padding + 3)
            rectangle(image, top_left, bottom_right, color_box, -1)

            # Draw text
            image = putText(image, text, (x, y), font, font_scale, color_text, thickness)

            # save stage image
            file_name = f"{image_name}_{self.stage_ctr:03}.png"

            # downsample image
            if downsample:
                image = resize(image, (0, 0), fx=0.5, fy=0.5)

            # append to images list
            self.calibration_stage_images.append(image)

            # print image
            if self.parent.default_settings['stage_images']:
                self.save_stage_image(image, file_name)
            first_image = True

    def do_camera_calibration(self, detected_points_2d=None, ground_truth_points_2d=None,
                              initial_positions_3d=None, use_ground_truth_points=False):
        # calibration settings
        self.use_img_ctr = self.parent.cali_vars['use_image_center'].get()
        self.use_ground_truth_points = use_ground_truth_points
        # calibration data
        self.detected_points_2d = detected_points_2d
        self.ground_truth_points_2d = ground_truth_points_2d
        self.initial_positions_3d = initial_positions_3d
        # initialize XYZ1
        self.XYZ1, self.xy1_detected = self.get_initial_positions_XYZ1()
        try:
            # initialize camera matrix and extrinsics
            self.initialization()
            self.print_stage_images()
            # Gauss-Newton optimization
            eta_vec_min = self.gauss_newton()
            # update eta
            self.update_optimization(eta_vec_min, mask_max_errors=True)
            # print stage image
            self.print_stage_images()
            # print to user
            self.results_to_messagebox()
            # write frames to video
            self.write_frames_to_video()
        except np.linalg.LinAlgError as e:
            messagebox.showerror("Optimization Error", f"Matrix inversion failed: {e}")
        except ValueError as e:
            messagebox.showerror("Optimization Error", f"Value error: {e}")
        except TypeError as e:
            messagebox.showerror("Optimization Error", f"Type error: {e}")
        except OverflowError as e:
            messagebox.showerror("Optimization Error", f"Overflow error: {e}")
        except Exception as e:
            messagebox.showerror("Optimization Error", f"An unexpected error occurred: {e}")
        # finally:
        #     self.parent.quit()

    def update_eta(self, eta_vec):
        for key, counter in zip(self.eta.keys(), range(len(eta_vec))):
            self.eta[key] = copy.deepcopy(eta_vec[counter][0])

    def update_optimization(self, eta_vec, mask_max_errors=False):
        # update eta
        self.update_eta(eta_vec)
        # update camera model
        self.update_camera_model()
        # project camera model
        self.project_camera_model()
        # update the Jacobian
        self.update_jacobian(mask_max_errors=mask_max_errors)
        # update objective function
        self.update_objective_function()

    def gauss_newton(self, alpha=0.40, trigger=500, mask_max_errors=False):
        # initialize
        max_iter, iter_ctr, tol, obj, obj_prev = 999, 0, 1e-7, 1e9, 1e9
        tol = 1e-16 if self.use_ground_truth_points else tol
        max_iter = 2222 if self.use_ground_truth_points else max_iter
        eta_vec_init = np.array(list(self.eta.values())).reshape(-1, 1)
        lm = np.eye(eta_vec_init.shape[0]) * 1e-12  # regularization
        # iterate
        while iter_ctr < max_iter:
            if self.objective_function < trigger:
                alpha = 1.0 - 0.5 * ((max_iter - iter_ctr) / max_iter) ** 2
            eta_vec = np.array(list(self.eta.values())).reshape(-1, 1)
            # optimize gauss newton
            try:
                inv_JtJ = inv(np.matmul(self.J.T, self.J) + lm)
            except np.linalg.LinAlgError as e:
                print(f"Matrix inversion failed: {e}")
                # update eta
                self.update_eta(alpha * eta_vec_min + (1.0 - alpha) * eta_vec_init)
                continue
            eta_vec_1 = eta_vec - np.matmul(inv_JtJ, np.matmul(self.J.T, self.D))
            # low pass filter
            eta_vec_low_pass = alpha * eta_vec + (1.0 - alpha) * eta_vec_1
            # update eta
            self.update_optimization(eta_vec_low_pass, mask_max_errors=mask_max_errors)
            # check convergence
            if obj > self.objective_function:
                obj = copy.deepcopy(self.objective_function)
                eta_vec_min = copy.deepcopy(eta_vec_low_pass)
                # output to user
                sys.stdout.write("\r" + f"min value objective function = {obj:.9f}, alpha = {alpha:.3f}")
            # print stage image
            if self.objective_function > trigger / 2:
                self.print_stage_images()
            elif iter_ctr % 100 == 0:
                self.print_stage_images()
            # check convergence
            if abs(obj_prev - self.objective_function) < tol:
                break
            elif abs(obj_prev - self.objective_function) < 1:
                mask_max_errors = True
            obj_prev = copy.deepcopy(self.objective_function)
            # update counter
            iter_ctr += 1
        return eta_vec_min

    def write_frames_to_video(self, fps=10):
        h, w = self.calibration_stage_images[0].shape[:2]
        # save video
        if self.parent.default_settings['stage_images_to_video']:
            file_name = f"{'calibration_optimization.mp4'}"
            # check if file exists
            if os.path.exists(file_name):
                os.remove(file_name)
            out = VideoWriter(file_name, VideoWriter_fourcc(*'DIVX'), fps, (w, h))
            for frame in self.calibration_stage_images:
                out.write(frame)
            # repeat last frame x7
            for _ in range(7):
                out.write(frame)
            out.release()

    def results_to_messagebox(self, ):
        lines = []
        if self.virtual_camera:
            lines.append('Calibration Results Versus Ground Truth')
            lines.append('---------------------------------------')
            lines.append('Camera Matrix')
            lines.append('focal length = (%.3f, %.3f)' % (self.eta['f'], self.eta_ground_truth['f']))
            lines.append('         \u03B1 = (%.3f, %.3f)' % (self.eta['lambda'], self.eta_ground_truth['lambda']))
            lines.append('         dX = (%.3f, %.3f)' % (self.eta['dX'], self.eta_ground_truth['dX']))
            lines.append('         dY = (%.3f, %.3f)' % (self.eta['dY'], self.eta_ground_truth['dY']))
            lines.append('---------------------------------------')
            lines.append('Lens Distortion')
            lines.append('k2 = (%.6f, %.6f)' % (self.eta['kappa_2'], self.eta_ground_truth['kappa_2']))
            lines.append('k3 = (%.6f, %.6f)' % (self.eta['kappa_3'], self.eta_ground_truth['kappa_3']))
            lines.append('k4 = (%.6f, %.6f)' % (self.eta['kappa_4'], self.eta_ground_truth['kappa_4']))
            lines.append('---------------------------------------')
            lines.append('Extrinsics')
            for key in self.eta:
                if 'image' in key and 'panel' in key:
                    image_label, panel_label, val = key.split('_')[0], key.split('_')[1], key.split('_')[2]
                    if val == 'roll' or val == 'pitch' or val == 'yaw':
                        num_measured = self.eta[key] * 180 / np.pi
                        num_ground_truth = self.eta_ground_truth[key] * 180 / np.pi
                    else:
                        num_measured = self.eta[key]
                        num_ground_truth = self.eta_ground_truth[key]
                    lines.append(panel_label + ' ' + val + ' = (%.3f, %.3f)' % (num_measured, num_ground_truth))
                    if val == 'tZ':
                        lines.append('---------------------------------------')

        else:
            lines.append('Calibration Results')
            lines.append('---------------------------------------')
            lines.append('Camera Matrix')
            lines.append('focal length = %.3f' % self.eta['f'])
            lines.append('         \u03B1 = %.3f' % self.eta['lambda'])
            lines.append('         dX = %.3f' % self.eta['dX'])
            lines.append('         dY = %.3f' % self.eta['dY'])
            lines.append('---------------------------------------')
            lines.append('Lens Distortion')
            lines.append('k2 = %.6f' % self.eta['kappa_2'])
            lines.append('k3 = %.6f' % self.eta['kappa_3'])
            lines.append('k4 = %.6f' % self.eta['kappa_4'])
            lines.append('---------------------------------------')
            lines.append('Extrinsics')
            for key in self.eta:
                if 'image' in key and 'panel' in key:
                    image_label, panel_label, val = key.split('_')[0], key.split('_')[1], key.split('_')[2]
                    if val == 'roll' or val == 'pitch' or val == 'yaw':
                        num_measured = self.eta[key] * 180 / np.pi
                    else:
                        num_measured = self.eta[key]
                    lines.append(image_label + ' ' + panel_label + ' ' + val + ' = %.3f' % num_measured)
                    if val == 'tZ':
                        lines.append('---------------------------------------')
        # print to user
        title, message = 'Calibration Results', '\n'.join(lines)
        if self.virtual_camera:
            messagebox.showinfo(title, message)
        else:
            self.show_scrollable_messagebox(title, message)

    def show_scrollable_messagebox(self, title, message):
        # Create a Toplevel window
        top = Toplevel()
        top.title(title)

        # Set fixed size for the Toplevel window
        top.geometry('400x300')

        # Create a frame for better styling
        frame = ttk.Frame(top, padding="10")
        frame.pack(fill='both', expand=True)
        frame.pack_propagate(False)  # Prevent frame from resizing

        # Create a Text widget with a Scrollbar
        text = Text(frame, wrap='word', font=("Helvetica", 10), relief='solid', borderwidth=1)
        text.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(frame, command=text.yview)
        scrollbar.pack(side='right', fill='y')

        text.config(yscrollcommand=scrollbar.set)

        # Insert the message into the Text widget
        text.insert('1.0', message)

        # Make the Text widget read-only
        text.config(state='disabled')

        # Add a frame for the button and icon
        button_frame = ttk.Frame(top, padding="10")
        button_frame.pack(fill='x', side='bottom')

        # Add the icon if provided
        if self.parent.icon_path:
            icon = tk.PhotoImage(file='.'.join(self.parent.icon_path.split('.')[:-1]) + '.png')
            # Resize the icon
            icon = icon.subsample(5, 5)
            icon_label = ttk.Label(button_frame, image=icon)
            icon_label.image = icon  # Keep a reference to the image
            icon_label.pack(side='left', padx=(0, 10))

        # Add a button to close the message box
        close_button = ttk.Button(button_frame, text="Close", command=top.destroy)
        close_button.pack(side='right')

    def return_eta_vector(self):
        return np.array(list(self.eta.values()))

    def update_jacobian(self, index_0=0, eTan=1e-6, gaussian_mask=False, mask_max_errors=False, error_threshold=0.5):
        # projected points
        xy_projected = self.unpack_dictionary(self.xy1_projected)[:, :2]
        # detected points
        xy_detected = self.unpack_dictionary(self.xy1_detected)[:, :2]
        # delta
        delta, K = xy_projected - xy_detected, self.camera_model['K']
        # initialize Jacobian
        J = OrderedDict()
        # project extrinsics
        self.project_extrinsics()
        # copy xyz
        xy1 = copy.deepcopy(self.xyz)
        xy1_dk2 = copy.deepcopy(self.xyz)
        xy1_dk3 = copy.deepcopy(self.xyz)
        xy1_dk4 = copy.deepcopy(self.xyz)
        for image_label in list(self.xyz):
            # per panel do
            for panel in list(self.xyz[image_label]):
                # apply lens distortion
                _lens_ = self.apply_lens_distortion_dX(self.xyz[image_label][panel])
                xy1[image_label][panel] = _lens_[0]
                xy1_dk2[image_label][panel] = _lens_[1]
                xy1_dk3[image_label][panel] = _lens_[2]
                xy1_dk4[image_label][panel] = _lens_[3]
        # derivatives: K matrix
        xy1_unpack = self.unpack_dictionary(xy1)
        D_df = (np.matmul(self.camera_model['K_df'], xy1_unpack.T).T[:, :2] * delta).sum(axis=1)
        D_dlambda = (np.matmul(self.camera_model['K_dlambda'], xy1_unpack.T).T[:, :2] * delta).sum(axis=1)
        D_ddX = (np.matmul(self.camera_model['K_ddX'], xy1_unpack.T).T[:, :2] * delta).sum(axis=1)
        D_ddY = (np.matmul(self.camera_model['K_ddY'], xy1_unpack.T).T[:, :2] * delta).sum(axis=1)
        # derivatives: lens distortion
        D_k2 = (np.matmul(self.camera_model['K'], self.unpack_dictionary(xy1_dk2).T).T[:, :2] * delta).sum(axis=1)
        D_k3 = (np.matmul(self.camera_model['K'], self.unpack_dictionary(xy1_dk3).T).T[:, :2] * delta).sum(axis=1)
        D_k4 = (np.matmul(self.camera_model['K'], self.unpack_dictionary(xy1_dk4).T).T[:, :2] * delta).sum(axis=1)
        # add to ordered dictionary
        J['D_df'] = D_df.reshape(-1, 1)
        J['D_dlambda'] = D_dlambda.reshape(-1, 1)
        J['D_ddX'] = D_ddX.reshape(-1, 1)
        J['D_ddY'] = D_ddY.reshape(-1, 1)
        J['D_dk2'] = D_k2.reshape(-1, 1)
        J['D_dk3'] = D_k3.reshape(-1, 1)
        J['D_dk4'] = D_k4.reshape(-1, 1)
        # extrinsics
        for image_label in list(self.XYZ1):
            # per panel do
            for panel in list(self.XYZ1[image_label]):
                # initialize
                columns, XYZ1 = np.zeros((delta.shape[0], 6)), self.XYZ1[image_label][panel]
                # normalize to surface of the sphere
                P = np.matmul(self.camera_model[image_label][panel]['Rt'], XYZ1.T).T
                # rotate and translate points
                XYZ_droll = np.matmul(self.camera_model[image_label][panel]['Rt_droll'], XYZ1.T).T
                XYZ_dpitch = np.matmul(self.camera_model[image_label][panel]['Rt_dpitch'], XYZ1.T).T
                XYZ_dyaw = np.matmul(self.camera_model[image_label][panel]['Rt_dyaw'], XYZ1.T).T
                XYZ_dtX = np.matmul(self.camera_model[image_label][panel]['Rt_dtX'], XYZ1.T).T
                XYZ_dtY = np.matmul(self.camera_model[image_label][panel]['Rt_dtY'], XYZ1.T).T
                XYZ_dtZ = np.matmul(self.camera_model[image_label][panel]['Rt_dtZ'], XYZ1.T).T
                # lens distorted coordinates
                xy1_droll = self.apply_lens_distortion_extrinsics_dX(P, XYZ_droll)
                xy1_dpitch = self.apply_lens_distortion_extrinsics_dX(P, XYZ_dpitch)
                xy1_dyaw = self.apply_lens_distortion_extrinsics_dX(P, XYZ_dyaw)
                xy1_dtX = self.apply_lens_distortion_extrinsics_dX(P, XYZ_dtX)
                xy1_dtY = self.apply_lens_distortion_extrinsics_dX(P, XYZ_dtY)
                xy1_dtZ = self.apply_lens_distortion_extrinsics_dX(P, XYZ_dtZ)
                # text lens distorted coordinates
                xy1_drollb = self.apply_lens_distortion_extrinsics_dXb(P, XYZ_droll)
                xy1_dpitchb = self.apply_lens_distortion_extrinsics_dXb(P, XYZ_dpitch)
                xy1_dyawb = self.apply_lens_distortion_extrinsics_dXb(P, XYZ_dyaw)
                xy1_dtXb = self.apply_lens_distortion_extrinsics_dXb(P, XYZ_dtX)
                xy1_dtYb = self.apply_lens_distortion_extrinsics_dXb(P, XYZ_dtY)
                xy1_dtZb = self.apply_lens_distortion_extrinsics_dXb(P, XYZ_dtZ)
                # check difference
                d1, d2 = np.abs(xy1_droll - xy1_drollb).max(), np.abs(xy1_dpitch - xy1_dpitchb).max()
                d3, d4 = np.abs(xy1_dyaw - xy1_dyawb).max(), np.abs(xy1_dtX - xy1_dtXb).max()
                d5, d6 = np.abs(xy1_dtY - xy1_dtYb).max(), np.abs(xy1_dtZ - xy1_dtZb).max()
                if d1 > eTan or d2 > eTan or d3 > eTan or d4 > eTan or d5 > eTan or d6 > eTan:
                    print("Arctan error: %.12f" % np.max([d1, d2, d3, d4, d5, d6]) + "\n")
                # camera coordinates
                xy1_droll = np.matmul(self.camera_model['K'], xy1_droll.T).T
                xy1_dpitch = np.matmul(self.camera_model['K'], xy1_dpitch.T).T
                xy1_dyaw = np.matmul(self.camera_model['K'], xy1_dyaw.T).T
                xy1_dtX = np.matmul(self.camera_model['K'], xy1_dtX.T).T
                xy1_dtY = np.matmul(self.camera_model['K'], xy1_dtY.T).T
                xy1_dtZ = np.matmul(self.camera_model['K'], xy1_dtZ.T).T
                # delta product
                D_droll = (xy1_droll[:, :2] * delta[index_0:index_0 + XYZ1.shape[0], :]).sum(axis=1)
                D_dpitch = (xy1_dpitch[:, :2] * delta[index_0:index_0 + XYZ1.shape[0], :]).sum(axis=1)
                D_dyaw = (xy1_dyaw[:, :2] * delta[index_0:index_0 + XYZ1.shape[0], :]).sum(axis=1)
                D_dtX = (xy1_dtX[:, :2] * delta[index_0:index_0 + XYZ1.shape[0], :]).sum(axis=1)
                D_dtY = (xy1_dtY[:, :2] * delta[index_0:index_0 + XYZ1.shape[0], :]).sum(axis=1)
                D_dtZ = (xy1_dtZ[:, :2] * delta[index_0:index_0 + XYZ1.shape[0], :]).sum(axis=1)
                # update columns
                columns[index_0:index_0 + XYZ1.shape[0], 0] = D_droll
                columns[index_0:index_0 + XYZ1.shape[0], 1] = D_dpitch
                columns[index_0:index_0 + XYZ1.shape[0], 2] = D_dyaw
                columns[index_0:index_0 + XYZ1.shape[0], 3] = D_dtX
                columns[index_0:index_0 + XYZ1.shape[0], 4] = D_dtY
                columns[index_0:index_0 + XYZ1.shape[0], 5] = D_dtZ
                index_0 = index_0 + XYZ1.shape[0]
                # add to Jacobian
                J['_'.join([image_label, panel])] = columns
        # Jacobian
        self.J = np.concatenate(list(J.values()), axis=1)
        # distance vector
        self.D = ((delta ** 2).sum(axis=1) * 0.5).reshape(-1, 1)
        if gaussian_mask:
            dist = xy_detected - np.array([self.camera_model['K'][0, -1], self.camera_model['K'][1, -1]]).reshape(1, -1)
            dist = np.linalg.norm(dist, axis=1)
            dist = dist / dist.max()
            dist = np.exp(np.sqrt(dist)).reshape(-1, 1)
            # dist = np.exp(1 / dist).reshape(-1, 1)
            self.D = self.D * dist
            self.J = self.J * dist
        if mask_max_errors:
            mask = (delta ** 2).sum(axis=1) < error_threshold
            self.D = self.D[mask]
            self.J = self.J[mask, :]

    def apply_lens_distortion_extrinsics_dX(self, P, dP):
        k2, k3, k4 = self.camera_model['kappa'][0], self.camera_model['kappa'][1], self.camera_model['kappa'][2]
        # NORMALIZATION FACTOR
        alpha = np.sqrt((P ** 2).sum(axis=1))
        # 2 sphere coordinates
        Px, Py, Pz = P[:, 0], P[:, 1], P[:, 2]
        qx, qy, qz = Px / alpha, Py / alpha, Pz / alpha
        phi, theta = np.arctan2(qy, qx), np.arccos(qz)
        # lens distortion
        radius = theta + k2 * theta ** 2 + k3 * theta ** 3 + k4 * theta ** 4
        # derivatives
        dPx, dPy, dPz = dP[:, 0], dP[:, 1], dP[:, 2]
        dP_P = (P * dP).sum(axis=1)
        dqx = dPx / alpha - (Px / alpha ** 3) * dP_P
        dqy = dPy / alpha - (Py / alpha ** 3) * dP_P
        dqz = dPz / alpha - (Pz / alpha ** 3) * dP_P
        d_theta = - 1 / np.sqrt(1 - qz ** 2) * dqz
        d_phi = (1 / (1 + qy ** 2 / qx ** 2)) * (dqy * qx - qy * dqx) / qx ** 2
        d_radius = d_theta * (1 + 2 * k2 * theta + 3 * k3 * theta ** 2 + 4 * k4 * theta ** 3)
        # derivative of lens distorted coordinates
        x = d_radius * np.cos(phi) - radius * np.sin(phi) * d_phi
        y = d_radius * np.sin(phi) + radius * np.cos(phi) * d_phi
        return np.array([x, y, np.zeros(len(x))]).T

    def apply_lens_distortion_extrinsics_dXb(self, P, dP):
        # lens parameters
        k2, k3, k4 = self.camera_model['kappa'][0], self.camera_model['kappa'][1], self.camera_model['kappa'][2]
        # 2 sphere coordinates
        Px, Py, Pz = P[:, 0], P[:, 1], P[:, 2]
        mag_2 = (Px ** 2 + Py ** 2 + Pz ** 2)
        # spherical angles
        phi, theta = np.arctan2(Py, Px), np.arctan2(np.sqrt(Px ** 2 + Py ** 2), Pz)
        # lens distortion
        r = theta + k2 * theta ** 2 + k3 * theta ** 3 + k4 * theta ** 4
        # derivatives
        dPx, dPy, dPz = dP[:, 0], dP[:, 1], dP[:, 2]
        d_phi = (Px * dPy - Py * dPx) / (Px ** 2 + Py ** 2)
        T1, T2 = (Px * dPx + Py * dPy) / (Pz * np.sqrt(Px ** 2 + Py ** 2)), np.sqrt(Px ** 2 + Py ** 2) * dPz / Pz ** 2
        d_theta = Pz ** 2 / mag_2 * (T1 - T2)
        d_r = d_theta * (1 + 2 * k2 * theta + 3 * k3 * theta ** 2 + 4 * k4 * theta ** 3)
        # derivative of lens distorted coordinates
        x = d_r * np.cos(phi) - r * np.sin(phi) * d_phi
        y = d_r * np.sin(phi) + r * np.cos(phi) * d_phi
        return np.array([x, y, np.zeros(len(x))]).T

    def apply_lens_distortion_dX(self, xyz, dTheta=True):
        xyz_copy = copy.deepcopy(xyz)
        k2, k3, k4 = self.camera_model['kappa'][0], self.camera_model['kappa'][1], self.camera_model['kappa'][2]
        # field angle
        field_angle = np.arccos(xyz_copy[:, 2])
        azi = np.arctan2(xyz_copy[:, 1], xyz_copy[:, 0])
        # field angle
        radius = field_angle + k2 * field_angle ** 2 + k3 * field_angle ** 3 + k4 * field_angle ** 4
        if dTheta:
            radius_dk2 = field_angle ** 2
            radius_dk3 = field_angle ** 3
            radius_dk4 = field_angle ** 4
        # pixel vector on the image plane
        xy1 = np.array([radius * np.cos(azi), radius * np.sin(azi), np.ones(len(field_angle), )])
        if dTheta:
            xy1_dk2 = np.array([radius_dk2 * np.cos(azi), radius_dk2 * np.sin(azi), np.zeros(len(field_angle), )])
            xy1_dk3 = np.array([radius_dk3 * np.cos(azi), radius_dk3 * np.sin(azi), np.zeros(len(field_angle), )])
            xy1_dk4 = np.array([radius_dk4 * np.cos(azi), radius_dk4 * np.sin(azi), np.zeros(len(field_angle), )])
            return xy1.T, xy1_dk2.T, xy1_dk3.T, xy1_dk4.T
        else:
            return xy1.T

    def project_lens_distortion_dX(self):
        self.xy1_lens_distorted = {}
        for image_label in list(self.XYZ1):
            # per panel do
            for panel in list(self.XYZ1[image_label]):
                # apply lens distortion
                xy1 = self.apply_lens_distortion_dX(self.xyz[image_label][panel])
                # add to dictionary
                if image_label not in self.xy1_lens_distorted:
                    self.xy1_lens_distorted[image_label] = {panel: copy.deepcopy(xy1)}
                else:
                    self.xy1_lens_distorted[image_label][panel] = copy.deepcopy(xy1)

    @staticmethod
    def unpack_dictionary(_dict_):
        _data_points_ = [value for sub_dict in _dict_.values() for value in sub_dict.values()]
        return np.concatenate(_data_points_)

    def update_objective_function(self):
        # objective function
        self.objective_function = self.D.sum()
        # print to user
        sys.stdout.write("\r")  # "\r" + " " * 80 + "\r"
        sys.stdout.write(f"objective function = {self.objective_function:.9f}")
        sys.stdout.flush()

    def project_camera_model(self, ):
        # project extrinsics
        self.project_extrinsics()
        # project lens distortion
        self.project_lens_distortion()
        # project camera matrix
        self.project_camera_matrix()

    def project_camera_matrix(self):
        self.xy1_projected = {}
        for image_label in list(self.XYZ1):
            # per panel do
            for panel in list(self.XYZ1[image_label]):
                # to camera coordinates
                xy1 = np.matmul(self.camera_model['K'], self.xy1_lens_distorted[image_label][panel].T).T
                # add to dictionary
                if image_label not in self.xy1_projected:
                    self.xy1_projected[image_label] = {panel: copy.deepcopy(xy1)}
                else:
                    self.xy1_projected[image_label][panel] = copy.deepcopy(xy1)

    def project_lens_distortion(self):
        self.xy1_lens_distorted = {}
        for image_label in list(self.XYZ1):
            # per panel do
            for panel in list(self.XYZ1[image_label]):
                # apply lens distortion
                xy1 = self.apply_lens_distortion(self.xyz[image_label][panel])
                # add to dictionary
                if image_label not in self.xy1_lens_distorted:
                    self.xy1_lens_distorted[image_label] = {panel: copy.deepcopy(xy1)}
                else:
                    self.xy1_lens_distorted[image_label][panel] = copy.deepcopy(xy1)

    def project_extrinsics(self, ):
        self.xyz = {}
        for image_label in list(self.XYZ1):
            # per panel do
            for panel in list(self.XYZ1[image_label]):
                # rotate and translate points
                XYZ = np.matmul(self.camera_model[image_label][panel]['Rt'], self.XYZ1[image_label][panel].T).T
                # normalize to surface of the sphere
                XYZ = XYZ / np.sqrt((XYZ ** 2).sum(axis=1)).reshape(-1, 1)
                # add to dictionary
                if image_label not in self.xyz:
                    self.xyz[image_label] = {panel: copy.deepcopy(XYZ)}
                else:
                    self.xyz[image_label][panel] = copy.deepcopy(XYZ)

    def apply_lens_distortion(self, xyz):
        k2, k3, k4 = self.camera_model['kappa'][0], self.camera_model['kappa'][1], self.camera_model['kappa'][2]
        # field angle
        field_angle = np.arccos(xyz[:, 2])
        azimuth_angle = np.arctan2(xyz[:, 1], xyz[:, 0])
        # field angle
        radius = field_angle + k2 * field_angle ** 2 + k3 * field_angle ** 3 + k4 * field_angle ** 4
        # pixel vector on the image plane
        xy1 = np.array([radius * np.cos(azimuth_angle),
                        radius * np.sin(azimuth_angle), np.ones(len(field_angle), )])
        return xy1.T

    def get_initial_positions_XYZ1(self):
        XYZ1, xy1 = {}, {}
        for image_ctr in range(len(self.detected_points_2d)):
            img_label = 'image%02d' % image_ctr
            # check if key exists
            if img_label not in XYZ1:
                XYZ1[img_label] = {}
                xy1[img_label] = {}
            for panel in self.detected_points_2d[image_ctr]:
                mask = copy.deepcopy(self.detected_points_2d[image_ctr][panel]['all_ids'])
                if len(mask) < 6:
                    continue
                # get points origin
                points_3d = copy.deepcopy(self.initial_positions_3d[panel][mask, :])
                homogeneous_coords = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
                XYZ1[img_label][panel] = homogeneous_coords
                # get detected points
                if self.use_ground_truth_points:
                    points_2d = copy.deepcopy(self.ground_truth_points_2d[image_ctr][panel][mask, :])
                else:
                    points_2d = copy.deepcopy(self.detected_points_2d[image_ctr][panel]['all_corners'])
                homogeneous_coords = np.hstack((points_2d, np.ones((points_2d.shape[0], 1))))
                xy1[img_label][panel] = homogeneous_coords
        return XYZ1, xy1

    def update_camera_model(self):
        eta = copy.deepcopy(self.eta)
        # camera matrix
        K = np.eye(3)
        x0, y0 = self.w / 2 - 0.5, self.h / 2 - 0.5
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = eta['f'], eta['f'] * eta['lambda'], x0 + eta['dX'], y0 + eta['dY']
        self.camera_model['K'] = K
        # derivatives
        self.camera_model['K_df'] = np.array([[1, 0, 0], [0, eta['lambda'], 0], [0, 0, 0]])
        self.camera_model['K_dlambda'] = np.array([[0, 0, 0], [0, eta['f'], 0], [0, 0, 0]])
        self.camera_model['K_ddX'] = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
        self.camera_model['K_ddY'] = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        # lens distortion
        kappa = np.array([eta['kappa_2'], eta['kappa_3'], eta['kappa_4']])
        self.camera_model['kappa'] = kappa
        # derivatives
        self.camera_model['kappa_d2'] = np.array([1, 0, 0])
        self.camera_model['kappa_d3'] = np.array([0, 1, 0])
        self.camera_model['kappa_d4'] = np.array([0, 0, 1])
        # extrinsics
        for key in eta:
            if 'image' in key and 'panel' in key:
                image_label, panel_label, var_label = key.split('_')[0], key.split('_')[1], key.split('_')[2]
                # register Rt, and the 6 derivatives only when var_label is roll
                if var_label == 'roll':
                    roll = eta['_'.join([image_label, panel_label, 'roll'])]
                    pitch = eta['_'.join([image_label, panel_label, 'pitch'])]
                    yaw = eta['_'.join([image_label, panel_label, 'yaw'])]
                    tX = eta['_'.join([image_label, panel_label, 'tX'])]
                    tY = eta['_'.join([image_label, panel_label, 'tY'])]
                    tZ = eta['_'.join([image_label, panel_label, 'tZ'])]
                    # get rotation matrix
                    rotor = self.rpy_to_rot_matrix(roll, pitch, yaw)
                    translation = np.array([tX, tY, tZ]).reshape(-1, 1)
                    Rt = np.concatenate((rotor, translation), axis=1)
                    # update derivatives
                    rotor_droll = self.rpy_to_rot_matrix(roll, pitch, yaw, dX='roll')
                    rotor_dpitch = self.rpy_to_rot_matrix(roll, pitch, yaw, dX='pitch')
                    rotor_dyaw = self.rpy_to_rot_matrix(roll, pitch, yaw, dX='yaw')
                    # assign
                    Rt_droll = np.concatenate((rotor_droll, np.zeros((3, 1))), axis=1)
                    Rt_dpitch = np.concatenate((rotor_dpitch, np.zeros((3, 1))), axis=1)
                    Rt_dyaw = np.concatenate((rotor_dyaw, np.zeros((3, 1))), axis=1)
                    Rt_dtX = np.concatenate((np.zeros((3, 3)), np.array([1, 0, 0]).reshape(-1, 1)), axis=1)
                    Rt_dtY = np.concatenate((np.zeros((3, 3)), np.array([0, 1, 0]).reshape(-1, 1)), axis=1)
                    Rt_dtZ = np.concatenate((np.zeros((3, 3)), np.array([0, 0, 1]).reshape(-1, 1)), axis=1)
                    # check exists
                    panel_label_cam_model = panel_label.replace('0', '_')
                    if image_label not in self.camera_model:
                        self.camera_model[image_label] = {panel_label_cam_model: {}}
                    if panel_label_cam_model not in self.camera_model[image_label]:
                        self.camera_model[image_label][panel_label_cam_model] = {}
                    # log all Rt
                    self.camera_model[image_label][panel_label_cam_model]['Rt'] = Rt
                    self.camera_model[image_label][panel_label_cam_model]['Rt_droll'] = Rt_droll
                    self.camera_model[image_label][panel_label_cam_model]['Rt_dpitch'] = Rt_dpitch
                    self.camera_model[image_label][panel_label_cam_model]['Rt_dyaw'] = Rt_dyaw
                    self.camera_model[image_label][panel_label_cam_model]['Rt_dtX'] = Rt_dtX
                    self.camera_model[image_label][panel_label_cam_model]['Rt_dtY'] = Rt_dtY
                    self.camera_model[image_label][panel_label_cam_model]['Rt_dtZ'] = Rt_dtZ
                else:
                    continue

    def initialize_optimization_variables(self, K, kappa, extrinsics):
        # initialize optimization variables
        h, w = self.parent.image['Virtual Camera'].height, self.parent.image['Virtual Camera'].width
        # Create an ordered dictionary
        eta = OrderedDict()
        # focal length
        eta['f'] = (K[0, 0] + K[1, 1]) / 2
        # focal length aspect ratio
        eta['lambda'] = 1.0
        # optical center offset dX
        eta['dX'] = K[0, 2] - (w / 2 - 0.5)
        # optical center offset dY
        eta['dY'] = K[1, 2] - (h / 2 - 0.5)
        # lens distortion coefficients
        eta['kappa_2'] = copy.deepcopy(kappa[0])
        # lens distortion coefficients
        eta['kappa_3'] = copy.deepcopy(kappa[1])
        # lens distortion coefficients
        eta['kappa_4'] = copy.deepcopy(kappa[2])
        # extrinsics
        for image_ctr in range(len(extrinsics)):
            for panel in extrinsics[image_ctr]:
                # variable names
                key_roll = 'image%02d_panel%02d_roll' % (image_ctr, int(panel.split('_')[-1]))
                key_pitch = 'image%02d_panel%02d_pitch' % (image_ctr, int(panel.split('_')[-1]))
                key_yaw = 'image%02d_panel%02d_yaw' % (image_ctr, int(panel.split('_')[-1]))
                key_tX = 'image%02d_panel%02d_tX' % (image_ctr, int(panel.split('_')[-1]))
                key_tY = 'image%02d_panel%02d_tY' % (image_ctr, int(panel.split('_')[-1]))
                key_tZ = 'image%02d_panel%02d_tZ' % (image_ctr, int(panel.split('_')[-1]))
                # assign
                eta[key_roll] = copy.deepcopy(extrinsics[image_ctr][panel]['roll_pitch_yaw'][0])
                eta[key_pitch] = copy.deepcopy(extrinsics[image_ctr][panel]['roll_pitch_yaw'][1])
                eta[key_yaw] = copy.deepcopy(extrinsics[image_ctr][panel]['roll_pitch_yaw'][2])
                eta[key_tX] = copy.deepcopy(extrinsics[image_ctr][panel]['translation'][0])
                eta[key_tY] = copy.deepcopy(extrinsics[image_ctr][panel]['translation'][1])
                eta[key_tZ] = copy.deepcopy(extrinsics[image_ctr][panel]['translation'][2])
        return eta
        # return copy.deepcopy(self.eta_ground_truth)

    def extrinsics_roll_pitch_yaw_from_rotor_translation(self, R, t):
        extrinsics = []
        for image_ctr in range(len(R)):
            _extrinsics_dict_ = {}
            for panel in R[image_ctr]:
                # initialize
                _extrinsics_dict_[panel] = {}
                # get roll pitch yaw
                rpy = self.rot_matrix_to_rpy(R[image_ctr][panel])
                # verify
                # rot = self.rpy_to_rot_matrix(rpy[0], rpy[1], rpy[2])
                # rot - R[image_ctr][panel]
                # get translation
                translation = t[image_ctr][panel].squeeze()
                # add to dictionary
                _extrinsics_dict_[panel] = {'roll_pitch_yaw': rpy, 'translation': translation}
            extrinsics.append(_extrinsics_dict_)
        return extrinsics

    def initialize_lens_distortion(self, K, R, t):
        # initialize distortion coefficients
        img_points_radius, world_points_theta = [], []
        for img_ctr in range(len(self.xy1_detected)):
            for panel in self.xy1_detected['image%02d' % img_ctr]:
                # homogeneous coordinates
                XYZ1 = self.XYZ1['image%02d' % img_ctr][panel]
                # get field angle of world points
                XYZ_world = np.matmul(np.concatenate((R[img_ctr][panel], t[img_ctr][panel]), axis=1), XYZ1.T).T
                XYZ_sphere = XYZ_world / np.sqrt((XYZ_world ** 2).sum(axis=1)).reshape(-1, 1)
                world_field_angle = np.arccos(XYZ_sphere[:, 2])
                # get detected points
                xy1 = copy.deepcopy(self.xy1_detected['image%02d' % img_ctr][panel])
                # get radial value of detected points
                xy1 = np.matmul(inv(K), xy1.T).T
                cam_coords_radius = np.sqrt((xy1[:, :2] ** 2).sum(axis=1))
                # append
                img_points_radius.append(cam_coords_radius)
                world_points_theta.append(world_field_angle)
        # linear least squares
        radius = np.concatenate(img_points_radius)
        theta = np.concatenate(world_points_theta)
        # M * kappa = radius - theta
        # M = [theta ** 2, theta ** 3, theta ** 4]
        M = np.array([theta ** 2, theta ** 3, theta ** 4]).T
        kappa = np.matmul(inv(np.matmul(M.T, M)), np.matmul(M.T, radius - theta))
        # check result
        # rect_error = np.array([radius- np.tan(theta)]).T
        # rectilinear_error = ((np.tan(theta) - radius) ** 2).sum()
        # M1 = np.matmul(np.array([theta, theta ** 2, theta ** 3, theta ** 4]).T,
        #                np.concatenate(([1], kappa)).reshape(-1, 1)).squeeze()
        # lens_distortion_error = ((M1 - radius) ** 2).sum()
        # kappa[0] = self.eta_ground_truth['kappa_2']
        # kappa[1] = self.eta_ground_truth['kappa_3']
        # kappa[2] = self.eta_ground_truth['kappa_4']
        return kappa

    def return_ground_truth_initialization(self):
        # debug ground truth
        K0 = self.parent.camera.return_camera_matrix()
        if np.logical_and(self.parent.int_vars['re-projection_mode'].get() == 'Apply',
                          self.parent.int_vars['re-projection_type'].get() == 'Rectilinear'):
            K0[0, 0] = K0[0, 0] / self.parent.default_settings['re-projection']['zoom_rectilinear']
            K0[1, 1] = K0[1, 1] / self.parent.default_settings['re-projection']['zoom_rectilinear']
        print('K0 = ')
        print(K0)
        # extrinsics ground truth
        R0, t0 = [], []
        for key in self.eta_ground_truth:
            if 'image' in key and 'panel' in key:
                image_label, panel_label, val = key.split('_')[0], key.split('_')[1], key.split('_')[2]
                if val == 'roll':
                    roll = self.eta_ground_truth[image_label + '_' + panel_label + '_roll']
                    pitch = self.eta_ground_truth[image_label + '_' + panel_label + '_pitch']
                    yaw = self.eta_ground_truth[image_label + '_' + panel_label + '_yaw']
                    tX = self.eta_ground_truth[image_label + '_' + panel_label + '_tX']
                    tY = self.eta_ground_truth[image_label + '_' + panel_label + '_tY']
                    tZ = self.eta_ground_truth[image_label + '_' + panel_label + '_tZ']
                    R0.append(self.rpy_to_rot_matrix(roll, pitch, yaw))
                    t0.append(np.array([tX, tY, tZ]).reshape(-1, 1))
                    # print('Rt = ' + image_label + ' ' + panel_label)
                    # print(np.concatenate((R0[0], t0[0]), axis=1))
                else:
                    continue
        # homography ground truth
        H_gt = []
        for i in range(len(R0)):
            H_gt.append(np.matmul(K0, np.concatenate((R0[i], t0[i]), axis=1)))
        return K0, R0, t0, H_gt

    def initialize_KRt(self, H=[], assume_right_angles=True):
        # DEBUG:
        if self.virtual_camera:
            self.K0, self.R0, self.t0, self.H_gt = self.return_ground_truth_initialization()
            self.H_gt = [self.H_gt[i] / self.H_gt[i][-1, -1] for i in range(3)][0]
        # per image do
        n_images, panels = len(self.xy1_detected), list(self.xy1_detected['image00'])
        for img_index in range(n_images):
            homo_per_panel = {}
            # per panel do:
            for panel in panels:
                # check
                if self.XYZ1['image%02d' % img_index][panel] is not None:
                    # get points
                    xy = self.xy1_detected['image%02d' % img_index][panel][:, :2]
                    XYZ = self.XYZ1['image%02d' % img_index][panel][:, :3]
                    # return the P matrix for each panel
                    homo_per_panel[panel] = self.rtn_homography_side(xy, XYZ).reshape(3, 3)
                else:
                    homo_per_panel[panel] = None
            H.append(homo_per_panel)
        # get camera matrix
        K = self.get_camera_matrix(H)
        # K00 = copy.deepcopy(K)
        # if self.use_img_ctr:
        #     K00[0, 2], K00[1, 2] = self.w / 2 - 0.5, self.h / 2 - 0.5
        # # extract all detected points per panel
        # xy1_detected = {image: [self.xy1_detected[image][panel] for panel in list(self.xy1_detected[image])]
        #                for image in list(self.xy1_detected)}
        # xy1_detected = {image: np.concatenate(xy1_detected[image]) for image in list(xy1_detected)}
        # xy1_camcoords = {image: np.matmul(inv(K00), xy1_detected[image].T).T for image in list(xy1_detected)}
        # # extract all origin points per panel
        # XYZ1 = {image: [self.XYZ1[image][panel] for panel in list(self.XYZ1[image])]
        #                for image in list(self.XYZ1)}
        # XYZ1 = {image: np.concatenate(XYZ1[image]) for image in list(XYZ1)}
        # # get the homography
        # H_graphy = {image: self.rtn_homography_side(xy1_camcoords[image], XYZ1[image]) for image in list(xy1_detected)}
        # R, t = [], []
        # # get extrinsics # loop through the images
        # for image in list(H_graphy):
        #     Homography, rotor, Rot, trans = H_graphy[image].reshape(3, 4), np.zeros((3, 3)), {}, {}
        #     r1, r2, r3, translation = Homography[:, 0], Homography[:, 1], Homography[:, 2], Homography[:, 3]
        #     # get the rotation and translation
        #     r1_norm, r2_norm, r3_norm = np.sqrt((r1 ** 2).sum()), np.sqrt((r2 ** 2).sum()), np.sqrt((r3 ** 2).sum())
        #     rotor[:, 0], rotor[:, 1], rotor[:, 2] = r1 / r1_norm, r2 / r2_norm, r3 / r3_norm
        #     # force orthonormal
        #     rotor[:, 0], rotor[:, 1], rotor[:, 2] = r1 / r1_norm, r2 / r2_norm, r3 / r3_norm
        #     rotation = self.quat_to_rot_matrix(self.rot_matrix_to_quat(rotor))
        #     # get translation
        #     translation = translation / np.array([r1_norm, r2_norm, r3_norm]).mean()
        #     # assign
        #     for panel in panels:
        #         Rot[panel], trans[panel] = rotation, translation.reshape(-1, 1)
        #     R.append(Rot)
        #     t.append(trans)
        # get extrinsics
        R, t = [], []
        for img_index in range(n_images):
            Rot, trans = {}, {}
            # per panel do:
            for panel in panels:
                if H[img_index][panel] is not None:
                    h1, h2, h3 = H[img_index][panel][:, 0], H[img_index][panel][:, 1], H[img_index][panel][:, 2]
                    translation = np.matmul(inv(K), h3.reshape(-1, 1))
                    Ra, Rb = np.zeros((3, 3)), np.zeros((3, 3))
                    if panel == 'panel_1':
                        r1, r2, translation = self.return_normalized_vectors(K, h1, h2, translation)
                        # rotation matrix
                        r3 = (np.cross(r1.T, r2.T) / np.sqrt((np.cross(r1.T, r2.T) ** 2).sum())).T
                        Ra[:, 0], Ra[:, 1], Ra[:, 2] = r1.squeeze(), r2.squeeze(), r3.squeeze()
                        Rb[:, 0], Rb[:, 1], Rb[:, 2] = r1.squeeze(), r2.squeeze(), -r3.squeeze()
                    elif panel == 'panel_2':
                        r2, r3, translation = self.return_normalized_vectors(K, h1, h2, translation)
                        # rotation matrix
                        r1 = (np.cross(r2.T, r3.T) / np.sqrt((np.cross(r2.T, r3.T) ** 2).sum())).T
                        Ra[:, 0], Ra[:, 1], Ra[:, 2] = r1.squeeze(), r2.squeeze(), r3.squeeze()
                        Rb[:, 0], Rb[:, 1], Rb[:, 2] = -r1.squeeze(), r2.squeeze(), r3.squeeze()
                    else:  # panel == 'panel_3':
                        r1, r3, translation = self.return_normalized_vectors(K, h1, h2, translation)
                        # rotation matrix
                        r2 = (np.cross(r1.T, r3.T) / np.sqrt((np.cross(r1.T, r3.T) ** 2).sum())).T
                        Ra[:, 0], Ra[:, 1], Ra[:, 2] = r1.squeeze(), r2.squeeze(), r3.squeeze()
                        Rb[:, 0], Rb[:, 1], Rb[:, 2] = r1.squeeze(), -r2.squeeze(), r3.squeeze()
                    # take rotor with max trace
                    rotation = Ra if (np.trace(Ra) > np.trace(Rb)) else Rb
                    # force orthonormal
                    rotation = self.quat_to_rot_matrix(self.rot_matrix_to_quat(rotation))
                    # append
                    Rot[panel] = rotation
                    trans[panel] = translation
            R.append(Rot)
            t.append(trans)
        # assume right angles
        if len(panels) > 1:
            if assume_right_angles:
                for img_index in range(len(R)):
                    R0, t0 = R[img_index]['panel_1'], t[img_index]['panel_1']
                    for panel in R[img_index]:
                        R[img_index][panel], t[img_index][panel] = R0, t0
        if self.use_img_ctr:
            K[0, 2], K[1, 2] = self.w / 2 - 0.5, self.h / 2 - 0.5
        return K, R, t

    @staticmethod
    def return_normalized_vectors(K, hA, hB, t):
        # get normalized vectors
        rA, rB = np.matmul(inv(K), hA.reshape(-1, 1)), np.matmul(inv(K), hB.reshape(-1, 1))
        # normalize
        rA_norm, rB_norm = np.sqrt((rA ** 2).sum()), np.sqrt((rB ** 2).sum())
        rA, rB = rA / rA_norm, rB / rB_norm
        # get translation
        t = t / np.array([rA_norm, rB_norm]).mean()
        return rA, rB, t

    @staticmethod
    def LLS_vs_SVD(M, SVD=False):
        if SVD:
            # write the singular value decomposition
            U, D, V_T = np.linalg.svd(M)
            SS = np.zeros(M.shape)
            SS[:M.shape[1], :M.shape[1]] = np.diag(D)
            # Omega = np.matmul(np.matmul(V_T.T, SS), np.matmul(SS, V_T))
            # get the homography vector
            p = V_T.T[:, -1] / V_T.T[-1, -1]
        else:
            # Write the singular valued equation M * p = 0, as B * q = b, where
            B, b = M[:, :-1], - M[:, -1]
            # solve the linear least squares solution
            q = np.matmul(inv(np.matmul(B.T, B)), np.matmul(B.T, b))
            # append
            p = np.array(list(q) + [1])
        return p

    def rtn_homography(self, xy, XY, x0=0, y0=0, use_img_ctr=True):
        if use_img_ctr:
            # assume the center of distortion is the image center
            x0 = self.parent.default_settings['sensor']['width'] / 2 - 0.5
            y0 = self.parent.default_settings['sensor']['height'] / 2 - 0.5
        # reference [Zhang's Method]: https://youtu.be/-9He7Nu3u8s
        # number of points
        n, x, y, X, Y = xy.shape[0], xy[:, 0] - x0, xy[:, 1] - y0, XY[:, 0], XY[:, 1]
        # create the A matrix
        A, aTx, aTy = np.zeros((2 * n, 9)), np.zeros((n, 9)), np.zeros((n, 9))
        # fill the A matrix
        aTx[:, 0], aTx[:, 1], aTx[:, 2] = X, Y, np.ones(n)
        aTx[:, 6], aTx[:, 7], aTx[:, 8] = -x * X, -x * Y, -x
        aTy[:, 3], aTy[:, 4], aTy[:, 5] = X, Y, np.ones(n)
        aTy[:, 6], aTy[:, 7], aTy[:, 8] = -y * X, -y * Y, -y
        # fill the A matrix
        A[:n, :], A[n:, :] = aTx, aTy
        # solve the system using LLS instead of SVD
        return self.LLS_vs_SVD(A)

    def rtn_homography_side(self, xy, XYZ, x0=0, y0=0):
        if self.use_img_ctr:
            # assume the center of distortion is the image center
            x0 = self.parent.default_settings['sensor']['width'] / 2 - 0.5
            y0 = self.parent.default_settings['sensor']['height'] / 2 - 0.5
        # reference [Zhang's Method]: https://youtu.be/-9He7Nu3u8s
        # number of points
        n, x, y, X, Y, Z = xy.shape[0], xy[:, 0] - x0, xy[:, 1] - y0, XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
        # create the A matrix
        A, aTx, aTy = np.zeros((2 * n, 12)), np.zeros((n, 12)), np.zeros((n, 12))
        # fill the A matrix
        aTx[:, 0], aTx[:, 1], aTx[:, 2], aTx[:, 3] = X, Y, Z, np.ones(n)
        aTx[:, 8], aTx[:, 9], aTx[:, 10], aTx[:, 11] = -x * X, -x * Y, -x * Z, -x
        aTy[:, 4], aTy[:, 5], aTy[:, 6], aTy[:, 7] = X, Y, Z, np.ones(n)
        aTy[:, 8], aTy[:, 9], aTy[:, 10], aTy[:, 11] = -y * X, -y * Y, -y * Z, -y
        # fill the A matrix
        A[:n, :], A[n:, :] = aTx, aTy
        # remove plane constraint
        if np.abs(np.diff(X)).max() < 1e-9:
            h_mask = [False, True, True, True, False, True, True, True, False, True, True, True]
        elif np.abs(np.diff(Y)).max() < 1e-9:
            h_mask = [True, False, True, True, True, False, True, True, True, False, True, True]
        elif np.abs(np.diff(Z)).max() < 1e-9:
            h_mask = [True, True, False, True, True, True, False, True, True, True, False, True]
        else:
            h_mask = [True, True, True, True, True, True, True, True, True, True, True, True]
        # H_gt = self.H_gt.flatten()[h_mask]
        H = self.LLS_vs_SVD(A[:, h_mask])
        return H  # np.matmul(A, H) # np.array([H, H_gt])

    def get_camera_matrix(self, H, Vec=[]):
        for image_n in range(len(H)):
            for panel in H[image_n]:
                II = H[image_n][panel]
                if II is not None:
                    Vec.append(self.homography_product_vector(II[:, 0], II[:, 1]))
                    Vec.append(self.homography_product_vector(II[:, 0], II[:, 0]) -
                               self.homography_product_vector(II[:, 1], II[:, 1]))
        # LLS vs SVD
        b_vec = self.LLS_vs_SVD(np.array(Vec))
        # normalize
        B = self.return_B_matrix(b_vec)
        # camera matrix
        K = inv(self.cholesky_decomposition(B))
        K = K / K[-1, -1]
        return K

    def return_B_matrix(self, b_vec):
        if self.use_img_ctr:
            b1, b2, b3 = b_vec[0], b_vec[1], b_vec[2]
            B = np.array([[b1, 0, 0], [0, b2, 0], [0, 0, b3]])
        else:
            b1, b2, b3, b4, b5 = b_vec[0], b_vec[1], b_vec[2], b_vec[3], b_vec[4]
            B = np.array([[b1, 0, b3],
                          [0, b2, b4],
                          [b3, b4, b5]])
        return B

    def homography_product_vector(self, hi, hj):
        if self.use_img_ctr:
            return np.array([hi[0] * hj[0], hi[1] * hj[1], hi[2] * hj[2]])
        else:
            return np.array([hi[0] * hj[0],
                             hi[1] * hj[1],
                             hi[2] * hj[0] + hi[0] * hj[2],
                             hi[2] * hj[1] + hi[1] * hj[2],
                             hi[2] * hj[2]])

    @staticmethod
    def cholesky_decomposition(A):
        """
        Perform Cholesky decomposition on a positive-definite matrix A.

        Parameters:
        A (np.ndarray): A positive-definite matrix.

        Returns:
        L (np.ndarray): A lower triangular matrix such that A = L * L.T
        """
        n = A.shape[0]
        L = np.zeros_like(A)

        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    L[i, j] = np.sqrt(A[i, i] - np.sum(L[i, :j] ** 2))
                else:
                    L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

        return L.T  # Return the upper triangular matrix

    # ROTATION MATRICES ################################################################################################

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

    def rpy_to_rot_matrix(self, roll, pitch, yaw, dX=None):
        """
        Convert roll, pitch, and yaw angles to a rotation matrix.

        Parameters:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.

        Returns:
        np.ndarray: A 3x3 rotation matrix.
        """
        # rotation matrices
        roll_mat = self.rotor_z(roll)
        pitch_mat = self.rotor_x(pitch)
        yaw_mat = self.rotor_y(yaw)
        # rotor
        if dX is None:
            return np.matmul(yaw_mat, np.matmul(pitch_mat, roll_mat))
        elif dX == 'roll':
            roll_mat_dX = self.rotor_dz(roll)
            return np.matmul(yaw_mat, np.matmul(pitch_mat, roll_mat_dX))
        elif dX == 'pitch':
            pitch_mat_dX = self.rotor_dx(pitch)
            return np.matmul(yaw_mat, np.matmul(pitch_mat_dX, roll_mat))
        elif dX == 'yaw':
            yaw_mat_dX = self.rotor_dy(yaw)
            return np.matmul(yaw_mat_dX, np.matmul(pitch_mat, roll_mat))

    @staticmethod
    def rot_matrix_to_rpy(R):
        """
        Convert a rotation matrix to roll, pitch, and yaw angles.

        Parameters:
        R (np.ndarray): A 3x3 rotation matrix.

        Returns:
        tuple: A tuple containing roll, pitch, and yaw angles in radians.
        """
        assert R.shape == (3, 3), "Rotation matrix must be 3x3"

        # Calculate pitch
        pitch = np.arcsin(-R[1, 2])

        # Check for gimbal lock
        if np.abs(np.cos(pitch)) > 1e-6:
            # Calculate roll and yaw
            roll = np.arctan2(R[1, 0], R[1, 1])
            yaw = np.arctan2(R[0, 2], R[2, 2])
        else:
            # Gimbal lock case
            yaw = np.arctan2(R[0, 1], R[2, 1])
            roll = 0

        return roll, pitch, yaw

    # write a function to convert a quaternion to a rotation matrix
    @staticmethod
    def quat_to_rot_matrix(q):
        """
        Convert a quaternion to a rotation matrix.

        Parameters:
        q (np.ndarray): A quaternion [w, x, y, z].

        Returns:
        np.ndarray: A 3x3 rotation matrix.
        :param q:
        :return:
        """
        w, x, y, z = q
        R = np.array([
            [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
        ])
        return R

    # write a function to convert a rotation matrix to a quaternion
    @staticmethod
    def rot_matrix_to_quat(R):
        """
        Convert a rotation matrix to a quaternion.

        Parameters:
        R (np.ndarray): A 3x3 rotation matrix.

        Returns:
        np.ndarray: A quaternion [w, x, y, z].
        """
        # Ensure the matrix is 3x3
        assert R.shape == (3, 3), "Rotation matrix must be 3x3"

        # Calculate the trace of the matrix
        tr = np.trace(R)

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2  # S=4*qw
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        norm = np.sqrt(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)
        return np.array([qw, qx, qy, qz]) / norm

    # ROLL PITCH YAW CONVENTIONS #######################################################################################

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

    # DERIVATIVES ######################################################################################################

    @staticmethod
    def rotor_dx(angle):  # pitch
        return np.array([
            [0, 0, 0],
            [0, -np.sin(angle), -np.cos(angle)],
            [0, np.cos(angle), -np.sin(angle)]
        ])

    @staticmethod
    def rotor_dy(angle):  # yaw
        return np.array([
            [-np.sin(angle), 0, np.cos(angle)],
            [0, 1, 0],
            [-np.cos(angle), 0, -np.sin(angle)]
        ])

    @staticmethod
    def rotor_dz(angle):  # roll
        return np.array([
            [-np.sin(angle), -np.cos(angle), 0],
            [np.cos(angle), -np.sin(angle), 0],
            [0, 0, 1]
        ])
