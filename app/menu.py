import customtkinter as ctk
from app.panels import *


class Menu(ctk.CTkTabview):
    def __init__(self, parent, pos_vars, ext_vars, int_vars, cali_vars, export_image):
        super().__init__(master=parent)
        self.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

        # tabs
        self.add('Extrinsics')
        self.add('Intrinsics')
        self.add('Calibration')
        self.add('Export')

        # widgets
        ExtrinsicsFrame(self.tab('Extrinsics'), pos_vars, ext_vars)
        IntrinsicsFrame(self.tab('Intrinsics'), int_vars)
        CalibrationFrame(self.tab('Calibration'), cali_vars)
        # EffectFrame(self.tab('Effects'), eff_vars)
        ExportFrame(self.tab('Export'), export_image)


class CalibrationFrame(ctk.CTkFrame):
    def __init__(self, parent, cali_vars):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(expand=True, fill='both')
        self.default_settings = self.master.master.master.default_settings
        # panel functions
        ChArUcoTarget(self, cali_vars)
        GeometricCalibrationPanel(self)


class ExtrinsicsFrame(ctk.CTkFrame):
    def __init__(self, parent, pos_vars, ext_vars, fixed_width=400):
        super().__init__(master=parent, fg_color='transparent', width=fixed_width)
        self.pack(expand=True, fill='both')
        self.default_settings = self.master.master.master.default_settings
        # panel functions
        EulerPanel(self, ext_vars)
        TranslationPanel(self, ext_vars)
        AxisAnglePanel(self, ext_vars)
        # SliderPanel(self, 'Rotation', pos_vars['rotate'], 0, 360)
        # SliderPanel(self, 'Zoom', pos_vars['zoom'], 0, 200)
        SegmentedPanel(self, 'Orientation', pos_vars['mirror_flip'], MIRROR_FLIP_OPTIONS)
        # revert button
        RevertButton(self, (pos_vars['mirror_flip'], MIRROR_FLIP_OPTIONS[0]),
                     (ext_vars['x'], self.default_settings['position']['x']),
                     (ext_vars['y'], self.default_settings['position']['y']),
                     (ext_vars['z'], self.default_settings['position']['z']),
                     (ext_vars['roll'], self.default_settings['euler']['roll']),
                     (ext_vars['pitch'], self.default_settings['euler']['pitch']),
                     (ext_vars['yaw'], self.default_settings['euler']['yaw']),
                     (ext_vars['field_angle'], self.default_settings['axis-angle']['field_angle']),
                     (ext_vars['azimuth'], self.default_settings['axis-angle']['azimuth']),
                     (ext_vars['rotation'], self.default_settings['axis-angle']['rotation']),
                     (ext_vars['rpy_convention'], self.default_settings['rpy_convention'][0]))
        # print(f'ExtrinsicsFrame width 1: {self.winfo_width()}')


class IntrinsicsFrame(ctk.CTkFrame):
    def __init__(self, parent, int_vars):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(expand=True, fill='both')
        self.default_settings = self.master.master.master.default_settings
        # panel functions
        CameraMatrixPanel(self, int_vars)
        LensDistortionPanel(self, int_vars)
        SensorTiltPanel(self, int_vars)
        ReProjectionPanel(self, int_vars)
        PlotLensGridPanel(self, int_vars)
        # revert button
        RevertButton(self,
                     (int_vars['focal_length'], self.default_settings['camera_matrix']['focal_length']),
                     (int_vars['dX'], self.default_settings['camera_matrix']['dX']),
                     (int_vars['dY'], self.default_settings['camera_matrix']['dY']),
                     (int_vars['yx_aspect_ratio'], self.default_settings['camera_matrix']['yx_aspect_ratio']),
                     (int_vars['lens_distortion'], self.default_settings['lens_distortion'][0]),
                     (int_vars['tilt_angle'], self.default_settings['sensor']['tilt_angle']),
                     (int_vars['tilt_azimuth'], self.default_settings['sensor']['tilt_azimuth']),
                     (int_vars['re-projection_type'], self.default_settings['re-projection']['type'][0]),
                     (int_vars['re-projection_mode'], self.default_settings['re-projection']['mode'][0]))
        # print(f'IntrinsicsFrame width 1: {self.winfo_width()}')


class EffectFrame(ctk.CTkFrame):
    def __init__(self, parent, eff_vars):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(expand=True, fill='both')

        DropDownPanel(self, eff_vars['effect'], EFFECT_OPTIONS)
        SliderPanel(self, 'Blur', eff_vars['blur'], 0, 30)
        SliderPanel(self, 'Contrast', eff_vars['contrast'], 0, 10)
        # revert button
        RevertButton(self, (eff_vars['effect'], EFFECT_OPTIONS[0]),
                     (eff_vars['blur'], BLUR_DEFAULT),
                     (eff_vars['contrast'], CONTRAST_DEFAULT))


class ExportFrame(ctk.CTkFrame):
    def __init__(self, parent, export_image):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(expand=True, fill='both')

        # data
        self.name_string = ctk.StringVar()
        self.file_string = ctk.StringVar(value='png')
        self.path_string = ctk.StringVar()

        # widgets
        FineNamePanel(self, self.name_string, self.file_string)
        FilePathPanel(self, self.path_string)
        SaveButton(self, export_image, self.name_string, self.file_string, self.path_string)
