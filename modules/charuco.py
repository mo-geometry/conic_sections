import numpy as np
from numpy.linalg import inv
import cv2
from cv2 import aruco
import tkinter.messagebox as messagebox
from fractions import Fraction
import logging


# ChAruco board variables
SQUARE_LENGTH = 1.0
MARKER_LENGTH = 0.8  # value < 1.0
BOARD_RESOLUTION = 2500


class CHARUCO:
    # constructor
    def __init__(self, *parent):
        # initialize parameters
        self.update_params(parent[0].cali_vars, parent[0].default_settings['charuco'])
        # initialize charuco board
        self.generate_board()

    # FUNCTIONS ########################################################################################################

    @staticmethod
    def read_chessboard(image, aruco_dict, board):
        """
        Charuco base pose estimation.
        https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/sandbox/ludovic/aruco_calibration_rotation.html
        """
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
        # DETECT POINTS
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
        all_corners, all_ids = [], []
        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize=(3, 3),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:
                all_corners = res2[1].squeeze()
                all_ids = res2[2].squeeze()
        all_corners = all_corners if len(all_corners) >= 6 else []
        all_ids = all_ids if len(all_corners) >= 6 else []
        return {'all_corners': all_corners, 'all_ids': all_ids}

    def write_detected_to_image(self, image, detected, panel):
        all_corners, all_ids = detected[panel]['all_corners'], detected[panel]['all_ids']
        if len(all_corners) > 0:
            for xy in all_corners:
                x, y = int(xy[0] + 0.5), int(xy[1] + 0.5)
                image = cv2.circle(image, (x, y), radius=2, color=(88, 255, 155), thickness=-1)
                image = cv2.circle(image, (x, y), radius=11, color=(33, 255, 255), thickness=3)
            for xy in self.projected_board_corners[panel][all_ids, :]:
                x, y = int(xy[0] + 0.5), int(xy[1] + 0.5)
                image = cv2.circle(image, (x, y), radius=7, color=(25, 225, 55), thickness=1)
        return image

    @staticmethod
    def write_error_metrics_to_image(image, reprojection_errors):
        all_errors = [reprojection_errors[p] for p in list(reprojection_errors)]
        all_errors = [x for x in all_errors if len(x) > 0]
        all_errors = np.concatenate(all_errors) if len(all_errors) > 0 else []
        if len(all_errors) > 0:
            # detection error
            E_mean, E_median = all_errors.mean(), np.median(all_errors)
            E_max, E_min = all_errors.max(), all_errors.min()
            image = cv2.putText(image, ' %d corners detected' % len(all_errors), (12, 62),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 33), 4)
            image = cv2.putText(image, ' => error (mean, median, max, min) = (%.3f, %.3f, %.3f, %.3f)'
                                % (E_mean, E_median, E_max, E_min),
                                (12, 108), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 33), 4)
        else:
            image = cv2.putText(image, ' 0 corners detected', (12, 62),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 33), 4)
        return image

    def return_reprojection_errors(self, detected, panel):
        all_corners, all_ids = detected[panel]['all_corners'], detected[panel]['all_ids']
        if len(all_corners) > 0:
            return np.sqrt(((self.projected_board_corners[panel][all_ids, :] - all_corners) ** 2).sum(axis=1))
        else:
            return []

    def update_corners_virtual_camera(self, image):
        """
        Update corners for virtual camera.
        :param image:
        :return: image
        """
        target_3d = np.logical_and(self.params['target_3d'].get(), self.params['ChArUco'].get())
        panels = ['panel_1', 'panel_2', 'panel_3'] if target_3d else ['panel_1']
        detected, reprojection_errors = {}, {}
        image_copy = image.copy()
        # Detect points
        for panel in panels:
            detected[panel] = self.read_chessboard(image_copy, self.aruco_dict[panel], self.board[panel])
            # Write corners to image for virtual camera.
            image = self.write_detected_to_image(image, detected, panel)
            # Reprojection errors
            reprojection_errors[panel] = self.return_reprojection_errors(detected, panel)
        # write statistics to image
        image = self.write_error_metrics_to_image(image, reprojection_errors)
        return image

    def generate_board(self):
        sq_x, sq_y = self.params['squares_x'].get(), self.params['squares_y'].get()
        self.board, self.aruco_dict, self.charuco_board = {}, {}, {}
        # https://stackoverflow.com/questions/75085270/cv2-aruco-charucoboard-create-not-found-in-opencv-4-7-0
        # generate charuco dictionaries
        for pan in ['panel_1', 'panel_2', 'panel_3']:
            self.aruco_dict[pan] = aruco.getPredefinedDictionary(getattr(aruco, self.params['dictionary_' + pan].get()))
        # generate charuco boards
        res_xy, sq_len, marker_len = self.return_image_resolution()
        self.board['panel_1'] = aruco.CharucoBoard((sq_x, sq_y), sq_len, marker_len, self.aruco_dict['panel_1'])
        self.board['panel_2'] = aruco.CharucoBoard((sq_x, sq_y), sq_len, marker_len, self.aruco_dict['panel_2'])
        self.board['panel_3'] = aruco.CharucoBoard((sq_x, sq_x), sq_len, marker_len, self.aruco_dict['panel_3'])
        # generate charuco boards
        for panel in ['panel_1', 'panel_2', 'panel_3']:
            try:
                res_xy = int((res_xy / sq_y) * sq_x) if np.logical_and(panel == 'panel_3', sq_y > sq_x) else res_xy
                self.charuco_board[panel] = self.board[panel].generateImage((res_xy, res_xy), None, 0, 1)
            except cv2.error as e:
                message = 'The number of markers exceeds the limit of the selected dictionary: ' + panel
                messagebox.showwarning('Marker Limit Exceeded', f'An unexpected error occurred: {e} \n' + message)
                return
            except Exception as e:
                messagebox.showwarning('Error', f'An unexpected error occurred: {e}' + '\n ' + panel)
                return
        # crop white space
        self.crop_white_space()
        # ground truth corners
        self.get_board_corners()
        # print
        # self.print_board()

    # SUBROUTINES ######################################################################################################

    def get_board_corners(self):
        self.board_corners, self.projected_board_corners, self.projected_board_corners_3d = {}, {}, {}
        sq_y, sq_x = self.params['squares_y'].get(), self.params['squares_x'].get()
        for panel in ['panel_1', 'panel_2', 'panel_3']:
            sq_y = sq_x if panel == 'panel_3' else sq_y
            h, w = self.charuco_board[panel].shape
            x0, y0 = w / 2 - 0.5, h / 2 - 0.5
            # x0, y0 = w / 2, h / 2
            scale = self.params['square_length_mm'].get() * sq_x / w
            # corner coordinates opencv convention
            X, Y = np.meshgrid(np.arange(0, w)[::int(w / sq_x)][1:], np.arange(0, h)[::int(h / sq_y)][1:])
            # get corners
            XYZ = np.ones((len(X.flatten()), 3))
            self.projected_board_corners[panel] = XYZ[:, :2]
            self.projected_board_corners_3d[panel] = XYZ[:, :]
            if panel == 'panel_1':  # [z = 0] plane
                XYZ[:, 0], XYZ[:, 1], XYZ[:, 2] = (X.flatten() - x0) * scale, (Y.flatten() - y0) * scale, XYZ[:, 2] * 0
            elif panel == 'panel_2':  # [x0 * scale] plane
                XYZ[:, 0] = XYZ[:, 0] * x0 * scale
                XYZ[:, 1], XYZ[:, 2] = (Y.flatten() - y0) * scale, - X.flatten() * scale
            elif panel == 'panel_3':  # [y0 * scale] plane
                XYZ[:, 1] = XYZ[:, 1] * (self.charuco_board['panel_1'].shape[0] / 2 - 0.5) * scale
                XYZ[:, 0], XYZ[:, 2] = (X.flatten() - x0) * scale, - Y.flatten() * scale
            self.board_corners[panel] = XYZ.copy()

    def return_image_resolution(self):
        # https://chev.me/arucogen/
        max_sq = max(self.params['squares_x'].get(), self.params['squares_y'].get())
        # board resolution
        res = int(BOARD_RESOLUTION - np.mod(BOARD_RESOLUTION, max_sq))
        ratio = Fraction(SQUARE_LENGTH / MARKER_LENGTH).limit_denominator()
        return res, ratio.numerator, ratio.denominator

    def crop_white_space(self):
        if self.params['crop_white_space']:
            sq_x, sq_y = self.params['squares_x'].get(), self.params['squares_y'].get()
            for panel in ['panel_1', 'panel_2', 'panel_3']:
                sq_y = sq_x if panel == 'panel_3' else sq_y
                h, w = self.charuco_board[panel].shape
                if sq_x == sq_y:
                    pass
                elif sq_x > sq_y:
                    mask = self.charuco_board[panel].astype(np.int32).sum(axis=1) - 255 * w != 0
                    self.charuco_board[panel] = self.charuco_board[panel][mask, :]
                else:  # sq_x < sq_y:
                    mask = self.charuco_board[panel].astype(np.int32).sum(axis=0) - 255 * h != 0
                    self.charuco_board[panel] = self.charuco_board[panel][:, mask]

    def print_board(self):
        if self.params['print_charuco']:
            for panel in ['panel_1', 'panel_2', 'panel_3']:
                cv2.imwrite('charuco_' + panel + '.png', self.charuco_board[panel])

    def update_params(self, cali_vars, defaults):
        self.params = cali_vars
        self.params['crop_white_space'] = defaults['crop_white_space']
        self.params['print_charuco'] = defaults['print_charuco']

    @staticmethod
    def draw_text(img, text,
                  font=cv2.FONT_HERSHEY_COMPLEX,
                  pos=(0, 0),
                  font_scale=3,
                  font_thickness=2,
                  text_color=(0, 255, 0),
                  text_color_bg=(0, 0, 0),
                  delta_y=5
                  ):

        x, y = pos[0], max(pos[1] - delta_y, 0)
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size[0], text_size[1] + 5 * delta_y
        img = cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
        img = cv2.putText(img, text, (x, int(y + text_h - 5 * delta_y + np.ceil(font_scale) - 1)),
                          font, font_scale, text_color, font_thickness)
        return img
