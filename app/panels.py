import customtkinter as ctk
from tkinter import filedialog
import numpy as np
import cv2
import os
from app.settings import *
from app.spinbox import *
from app.throttled_var import *
from modules.matplotlib_class import *
import sys
import tkinter.messagebox as messagebox
import copy

print(sys.version)
print(sys.executable)


class Panel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.pack(fill='x', pady=4, ipady=8)


class SliderPanel(Panel):
    def __init__(self, parent, text, data_var, min_value, max_value):
        super().__init__(parent=parent)

        # layout
        self.rowconfigure((0, 1), weight=1)
        self.columnconfigure((0, 1), weight=1)

        self.data_var = data_var
        # self.data_var.trace('w', self.update_text)

        ctk.CTkLabel(self, text=text).grid(column=0, row=0, sticky='W', padx=5)
        self.num_label = ctk.CTkLabel(self, text=self.data_var.get())
        self.num_label.grid(column=1, row=0, sticky='E', padx=5)

        ctk.CTkSlider(self, fg_color=SLIDER_BG, variable=self.data_var, command=self.update_text,
                      from_=min_value, to=max_value).grid(row=1, column=0, columnspan=2, sticky='EW', padx=5, pady=5)

    def update_text(self, *args):
        self.num_label.configure(text=f'{round(self.data_var.get(), 2, )}')


class SegmentedPanel(Panel):
    def __init__(self, parent, text, data_var, options):
        super().__init__(parent=parent)

        ctk.CTkLabel(self, text=text).pack()
        ctk.CTkSegmentedButton(self, variable=data_var, values=options).pack(expand=True, fill='both', padx=4, pady=4)


class SwitchPanel(Panel):
    def __init__(self, parent, *args):  # ((var, text), (var, text), (var, text), (var, text))
        super().__init__(parent=parent)

        for var, text, in args:
            switch = ctk.CTkSwitch(self, text=text, variable=var, button_color=BLUE, fg_color=SLIDER_BG,
                                   command=lambda: self.press_switch)
            switch.pack(side='left', expand=True, fill='both', padx=5, pady=5)

    def press_switch(self):
        # print('switch pressed')
        pass


class FineNamePanel(Panel):
    def __init__(self, parent, name_string, file_string):
        super().__init__(parent=parent)
        # Title Label
        self.title_label = ctk.CTkLabel(self, text="File Name", font=("Times", 16))
        self.title_label.pack(fill='x', padx=10, pady=5)
        # data
        self.name_string = name_string  # name space => name_space
        self.name_string.trace('w', self.update_text)
        self.file_string = file_string
        # check boxes for file format
        ctk.CTkEntry(self, textvariable=self.name_string).pack(fill='x', padx=20, pady=5)
        frame = ctk.CTkFrame(self, fg_color='transparent')
        jpg_check = ctk.CTkCheckBox(frame, text='jpg', variable=self.file_string, command=lambda: self.click('jpg'),
                                    onvalue='jpg', offvalue='png')
        png_check = ctk.CTkCheckBox(frame, text='png', variable=self.file_string, command=lambda: self.click('png'),
                                    onvalue='png', offvalue='jpg')
        jpg_check.pack(side='left', fill='x', expand=True, padx=25)
        png_check.pack(side='left', fill='x', expand=True, padx=5)
        frame.pack(expand=True, fill='x', padx=2, pady=(0, 5))
        # preview text
        self.output = ctk.CTkLabel(self, text='')
        self.output.pack(pady=(0, 5))

    def click(self, value):
        self.file_string.set(value)
        self.update_text()

    def update_text(self, *args):
        if self.name_string.get():
            text = self.name_string.get().replace(' ', '_') + '.' + self.file_string.get()
            self.output.configure(text=text)


class FilePathPanel(Panel):
    def __init__(self, parent, path_string):
        super().__init__(parent=parent)
        self.path_string = path_string
        ctk.CTkButton(self, text='Open Explorer', command=self.open_file_dialog).pack(pady=5)
        ctk.CTkEntry(self, textvariable=self.path_string).pack(expand=True, fill='both', padx=5, pady=5)

    def open_file_dialog(self):
        self.path_string.set(filedialog.askdirectory())


class DropDownPanel(ctk.CTkOptionMenu):
    def __init__(self, parent, data_var, options):
        super().__init__(master=parent, values=options, fg_color=DARK_GREY, button_color=DROPDOWN_MAIN_COLOUR,
                         button_hover_color=DROPDOWN_HOVER_COLOUR, dropdown_fg_color=DROPDOWN_MENU_COLOUR,
                         variable=data_var)
        self.pack(fill='x', pady=4)


class RevertButton(ctk.CTkButton):
    def __init__(self, parent, *args):
        super().__init__(master=parent, text='Revert', command=self.revert)
        self.pack(side='bottom', pady=10)
        self.args = args

    def revert(self):
        for var, value in self.args:
            if isinstance(var, ThrottledVar):
                var.var.set(value)
            else:
                var.set(value)


class SaveButton(ctk.CTkButton):
    def __init__(self, parent, export_image, name_string, file_string, path_string):
        super().__init__(master=parent, text='save', command=self.save)
        self.pack(side='bottom', pady=10)
        self.export_image = export_image
        self.name_string = name_string
        self.file_string = file_string
        self.path_string = path_string

    def save(self):
        self.export_image(self.name_string.get(), self.file_string.get(), self.path_string.get())


# INTRINSICS ###########################################################################################################


class CameraMatrixPanel(ctk.CTkFrame):
    def __init__(self, parent, camera_var, slider_width=125, text_box_width=55):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.pack(fill='both', pady=4, ipady=8)  # expand=True,
        # Camera Matrix variables
        self.focal_length_var = camera_var['focal_length']
        self.optical_center_dX_var = camera_var['dX']
        self.optical_center_dY_var = camera_var['dY']
        self.yx_aspect_ratio_var = camera_var['yx_aspect_ratio']

        # column configure
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        # Create title label and place it in the first row
        ctk.CTkLabel(self, text="Camera Matrix", justify='left').grid(row=0, column=0)
        ctk.CTkLabel(self, text="             ", justify='left').grid(row=0, column=1)
        ctk.CTkLabel(self, text="             ", justify='left').grid(row=0, column=2)

        # Create StringVar variables for focal length, optical center and aspect ratio and set their initial values
        ctk.CTkLabel(self, text="Focal Length (pxls)", justify='left', padx=10).grid(row=1, column=0)
        spinbox = FloatSpinbox(self, width=130, step_size=5.0, z_var=self.focal_length_var, min_value=200)
        spinbox.grid(row=1, column=1, columnspan=2)
        # Optical center
        ctk.CTkLabel(self, text="Optical ctr dX, dY", justify='left', padx=10).grid(row=2, column=0)
        self.dX_var_label = ctk.CTkEntry(self, textvariable=self.optical_center_dX_var,
                                         width=text_box_width, border_width=2, corner_radius=10)
        self.dY_var_label = ctk.CTkEntry(self, textvariable=self.optical_center_dY_var,
                                         width=text_box_width, border_width=2, corner_radius=10)
        self.dX_var_label.grid(row=2, column=1)
        self.dY_var_label.grid(row=2, column=2)
        # Aspect ratio
        ctk.CTkLabel(self, text="Y/X aspect ratio", justify='left', padx=10).grid(row=3, column=0)
        self.aspect_ratio_label = ctk.CTkEntry(self, textvariable=self.yx_aspect_ratio_var,
                                               width=2.05 * text_box_width, border_width=2, corner_radius=10)
        self.aspect_ratio_label.grid(row=3, column=1, columnspan=2)


class LensDistortionPanel(ctk.CTkFrame):
    def __init__(self, parent, camera_var, slider_width=125, text_box_width=55):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.pack(fill='both', pady=4, ipady=8)  # expand=True,
        # Lens Distortion Variables
        self.lens_distortion_var = camera_var['lens_distortion']
        radio_vars = self.master.default_settings['lens_distortion']

        # column configure
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # Create title label and place it in the first row
        ctk.CTkLabel(self, text="Lens Distortion", justify='center').grid(row=0, column=0)
        ctk.CTkLabel(self, text="               ", justify='center').grid(row=0, column=1)
        # Lens Distortion radio buttons
        self.radio1 = ctk.CTkRadioButton(self, text=radio_vars[0], variable=self.lens_distortion_var,
                                         value=radio_vars[0], command=self.radio_event)
        self.radio2 = ctk.CTkRadioButton(self, text=radio_vars[1], variable=self.lens_distortion_var,
                                         value=radio_vars[1], command=self.radio_event)
        self.radio3 = ctk.CTkRadioButton(self, text=radio_vars[2], variable=self.lens_distortion_var,
                                         value=radio_vars[2], command=self.radio_event)
        self.radio4 = ctk.CTkRadioButton(self, text=radio_vars[3], variable=self.lens_distortion_var,
                                         value=radio_vars[3], command=self.radio_event)
        self.radio5 = ctk.CTkRadioButton(self, text=radio_vars[4], variable=self.lens_distortion_var,
                                         value=radio_vars[4], command=self.radio_event)
        self.radio1.grid(row=1, column=0, padx=5, pady=5)
        self.radio2.grid(row=1, column=1, padx=5, pady=5)
        self.radio3.grid(row=2, column=0, padx=5, pady=5)
        self.radio4.grid(row=2, column=1, padx=5, pady=5)
        self.radio5.grid(row=3, column=0, padx=5, pady=5)

    def radio_event(self):
        # print(f"{self.lens_distortion_var.get()} selected")
        pass


class SensorTiltPanel(ctk.CTkFrame):
    def __init__(self, parent, camera_var, slider_width=125, text_box_width=55):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.pack(fill='both', pady=4, ipady=8)  # expand=True,
        # Sensor Tilt Variables
        self.tilt_angle_var = camera_var['tilt_angle']
        self.azimuth_var = camera_var['tilt_azimuth']

        # self.azimuth_var.trace('w', self.update_azimuth)
        # self.azimuth_var.var.trace('w', self.update_azimuth)  # throttled variable

        # column configure
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        # Create title label and place it in the first row
        ctk.CTkLabel(self, text="Sensor Tilt", justify='left').grid(row=0, column=0)
        ctk.CTkLabel(self, text="           ", justify='left').grid(row=0, column=1)
        ctk.CTkLabel(self, text="           ", justify='left').grid(row=0, column=2)

        # Create StringVar variables for tilt angle and azimuth and set their initial values
        ctk.CTkLabel(self, text="Tilt Angle (deg)", justify='left', padx=10).grid(row=1, column=0)
        spinbox = FloatSpinbox(self, width=130, step_size=0.5, z_var=self.tilt_angle_var, min_value=0, max_value=10)
        spinbox.grid(row=1, column=1, columnspan=2)
        ctk.CTkLabel(self, text="Azimuth (deg)", justify='left', padx=10).grid(row=2, column=0)
        ctk.CTkSlider(self, fg_color=SLIDER_BG, variable=self.azimuth_var, from_=-180, to=180,
                      command=self.update_azimuth, width=slider_width).grid(row=2, column=1)
        self.azimuth_value_label = ctk.CTkLabel(self, text=self.azimuth_var.get(), width=text_box_width)
        self.azimuth_value_label.grid(row=2, column=2)

    def update_azimuth(self, *args):
        self.azimuth_value_label.configure(text=f'{round(self.azimuth_var.get(), 1)}')


class ReProjectionPanel(ctk.CTkFrame):
    def __init__(self, parent, camera_var):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.pack(fill='both', pady=4, ipady=8)  # expand=True,
        # Sensor Tilt Variables
        self.re_projection_type_var = camera_var['re-projection_type']
        self.re_projection_mode_var = camera_var['re-projection_mode']
        radio_type_vars = self.master.default_settings['re-projection']['type']
        self.radio_mode_vars = self.master.default_settings['re-projection']['mode']

        # Create variables for each mode
        self.mode_var1 = ctk.BooleanVar(value=True)
        self.mode_var2 = ctk.BooleanVar(value=False)

        # column configure
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)

        # Create title label and place it in the first row
        ctk.CTkLabel(self, text="Re-projection", justify='left').grid(row=0, column=0, columnspan=2)
        # ctk.CTkLabel(self, text="             ", justify='left').grid(row=0, column=1)
        ctk.CTkLabel(self, text="             ", justify='left').grid(row=0, column=2)
        ctk.CTkLabel(self, text="             ", justify='left').grid(row=0, column=3)

        # ReProjection radio buttons
        self.radio1 = ctk.CTkRadioButton(self, text=radio_type_vars[0], variable=self.re_projection_type_var,
                                         value=radio_type_vars[0], command=self.radio_event)
        self.radio2 = ctk.CTkRadioButton(self, text=radio_type_vars[1], variable=self.re_projection_type_var,
                                         value=radio_type_vars[1], command=self.radio_event)
        self.radio3 = ctk.CTkRadioButton(self, text=radio_type_vars[2], variable=self.re_projection_type_var,
                                         value=radio_type_vars[2], command=self.radio_event)
        self.radio4 = ctk.CTkRadioButton(self, text=radio_type_vars[3], variable=self.re_projection_type_var,
                                         value=radio_type_vars[3], command=self.radio_event)
        self.radio1.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.radio2.grid(row=1, column=2, columnspan=2, padx=5, pady=5)
        self.radio3.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        self.radio4.grid(row=2, column=2, columnspan=2, padx=5, pady=5)

        # Create a switch panel with a switch for each mode
        switch_1 = ctk.CTkSwitch(self, text=self.radio_mode_vars[0], command=self.update_mode1,
                                 variable=self.mode_var1, button_color=BLUE, fg_color=SLIDER_BG)
        switch_2 = ctk.CTkSwitch(self, text=self.radio_mode_vars[1], command=self.update_mode2,
                                 variable=self.mode_var2, button_color=BLUE, fg_color=SLIDER_BG)
        switch_1.grid(row=3, column=0, columnspan=2, padx=10)
        switch_2.grid(row=3, column=2, columnspan=2, padx=10)

    def radio_event(self):
        # print(f"{self.re_projection_type_var.get()} selected")
        pass

    def update_mode1(self, *args):
        if self.mode_var1.get():
            self.mode_var2.set(False)
            self.re_projection_mode_var.set(self.radio_mode_vars[0])
            # print(self.re_projection_mode_var.get())
        else:
            self.mode_var2.set(True)
            self.re_projection_mode_var.set(self.radio_mode_vars[1])
            # print(self.re_projection_mode_var.get())

    def update_mode2(self, *args):
        if self.mode_var2.get():
            self.mode_var1.set(False)
            self.re_projection_mode_var.set(self.radio_mode_vars[1])
            # print(self.re_projection_mode_var.get())
        else:
            self.mode_var1.set(True)
            self.re_projection_mode_var.set(self.radio_mode_vars[0])
            # print(self.re_projection_mode_var.get())


# EXTRINSICS ###########################################################################################################


class EulerPanel(ctk.CTkFrame):
    def __init__(self, parent, object_var, slider_width=125, text_box_width=55):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.pack(fill='both', pady=4, ipady=8)  # expand=True,
        self.step_size = 0.25  # Define the step size
        # self.grid(sticky='nsew')

        self.roll_var = object_var['roll']
        self.pitch_var = object_var['pitch']
        self.yaw_var = object_var['yaw']
        # self.roll_var.var.trace('w', self.update_roll)        # throttled variable
        # self.pitch_var.var.trace('w', self.update_pitch)      # throttled variable
        # self.yaw_var.var.trace('w', self.update_yaw)          # throttled variable
        # self.roll_var.trace('w', self.update_roll)
        # self.pitch_var.trace('w', self.update_pitch)
        # self.yaw_var.trace('w', self.update_yaw)

        # Create StringVar for dropdown and slider
        # self.rpy_convention = tk.StringVar(value='RPY')
        # self.roll_var = tk.DoubleVar()
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)
        self.columnconfigure(4, weight=1)
        self.columnconfigure(5, weight=1)
        # First row
        ctk.CTkLabel(self, text="Euler Convention", padx=20).grid(row=0, column=0, columnspan=3)
        # ctk.CTkOptionMenu(self, variable=self.rpy_convention.get(), values=['RPY', 'YPR'])
        rpy_options = self.master.master.master.master.default_settings['rpy_convention']
        ctk.CTkOptionMenu(self, values=rpy_options,
                          fg_color=MID_GREY, button_color=DROPDOWN_MAIN_COLOUR,
                          button_hover_color=DROPDOWN_HOVER_COLOUR, dropdown_fg_color=DROPDOWN_MENU_COLOUR,
                          variable=object_var['rpy_convention']).grid(row=0, column=3, columnspan=3)

        # Second row
        ctk.CTkLabel(self, text="Roll", padx=5).grid(row=1, column=0)
        ctk.CTkLabel(self, text="Pitch", padx=5).grid(row=2, column=0)
        ctk.CTkLabel(self, text="Yaw", padx=5).grid(row=3, column=0)
        ctk.CTkSlider(self, fg_color=SLIDER_BG, variable=self.roll_var, from_=-55, to=55,
                      command=self.update_roll, width=slider_width).grid(row=1, column=1, columnspan=4)
        ctk.CTkSlider(self, fg_color=SLIDER_BG, variable=self.pitch_var, from_=-55, to=55,
                      command=self.update_pitch, width=slider_width).grid(row=2, column=1, columnspan=4)
        ctk.CTkSlider(self, fg_color=SLIDER_BG, variable=self.yaw_var, from_=-55, to=55,
                      command=self.update_yaw, width=slider_width).grid(row=3, column=1, columnspan=4)
        self.roll_value_label = ctk.CTkLabel(self, text=self.roll_var.get(), width=text_box_width)
        self.pitch_value_label = ctk.CTkLabel(self, text=self.pitch_var.get(), width=text_box_width)
        self.yaw_value_label = ctk.CTkLabel(self, text=self.yaw_var.get(), width=text_box_width)
        self.roll_value_label.grid(row=1, column=5)
        self.pitch_value_label.grid(row=2, column=5)
        self.yaw_value_label.grid(row=3, column=5)

    def update_roll(self, *args):
        # self.roll_var.set(round(self.roll_var.get() / self.step_size) * self.step_size)     # causes double update
        self.roll_value_label.configure(text=f'{round(self.roll_var.get(), 1)}')

    def update_pitch(self, *args):
        # self.pitch_var.set(round(self.pitch_var.get() / self.step_size) * self.step_size)     # causes double update
        self.pitch_value_label.configure(text=f'{round(self.pitch_var.get(), 1)}')

    def update_yaw(self, *args):
        # self.yaw_var.set(round(self.yaw_var.get() / self.step_size) * self.step_size)     # causes double update
        self.yaw_value_label.configure(text=f'{round(self.yaw_var.get(), 1)}')


class TranslationPanel(ctk.CTkFrame):
    def __init__(self, parent, object_var):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.pack(fill='both', pady=4, ipady=8)

        # column configure
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        # Create title label and place it in the first row
        ctk.CTkLabel(self, text="Translation", padx=20).grid(row=0, column=0, columnspan=2)

        # Create StringVar variables for x and y coordinates and set their initial values
        self.x_var = object_var['x']
        self.y_var = object_var['y']
        self.z_var = object_var['z']

        # Create labels for x and y coordinates
        ctk.CTkLabel(self, text="  [X, Y] (mm):").grid(row=1, column=0, padx=10)
        ctk.CTkLabel(self, text="   Z     (mm):").grid(row=2, column=0, padx=10)

        # Create entry boxes for x and y coordinates and place them using the place method
        x_entry = ctk.CTkEntry(self, textvariable=self.x_var, width=65, border_width=2, corner_radius=10)
        y_entry = ctk.CTkEntry(self, textvariable=self.y_var, width=65, border_width=2, corner_radius=10)

        x_entry.grid(row=1, column=1, padx=10)
        y_entry.grid(row=1, column=2, padx=1)

        spinbox = FloatSpinbox(self, width=150, step_size=3, z_var=self.z_var)
        spinbox.grid(row=2, column=1, columnspan=2)


class PlotLensGridPanel(ctk.CTkFrame):
    def __init__(self, parent, int_vars):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.pack(fill='both', pady=4, ipady=2)

        # Create 'Plot Lens Grid' button
        self.plot_button = ctk.CTkButton(self, text='Plot Lens Grid', command=self.plot_lens_grid)
        self.plot_button.pack(pady=10)

    def plot_lens_grid(self):
        # Placeholder for the actual plotting logic
        self.master.master.master.master.plotting.plot_lens_grid()


class AxisAnglePanel(ctk.CTkFrame):
    def __init__(self, parent, object_var, slider_width=80, value_width=55, text_width=55):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.pack(fill='both', pady=4, ipady=8)

        self.field_angle = object_var['field_angle']
        self.azimuth = object_var['azimuth']
        self.rotation = object_var['rotation']

        # Trace changes in x_var and y_var
        # self.field_angle.trace('w', self.update_field_angle)
        # self.azimuth.var.trace('w', self.update_azimuth)
        # self.rotation.var.trace('w', self.update_rotation)

        # column configure
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)

        # Create title labels
        ctk.CTkLabel(self, text="Axis Angle Quaternion", padx=20, justify='left').grid(row=0, column=0, columnspan=2)
        ctk.CTkLabel(self, text="Field Angle", anchor='w', padx=20, width=text_width).grid(row=1, column=0)
        ctk.CTkLabel(self, text="Azimuth", padx=20, anchor='w', width=text_width).grid(row=2, column=0)
        ctk.CTkLabel(self, text="Rotation", anchor='w', padx=20, width=text_width).grid(row=3, column=0)

        spinbox = FloatSpinbox(self, width=130, step_size=2.5, z_var=self.field_angle, min_value=0)
        spinbox.grid(row=1, column=1, columnspan=3)
        ctk.CTkSlider(self, fg_color=SLIDER_BG, variable=self.azimuth, from_=-180, to=180, width=slider_width,
                      command=self.update_azimuth).grid(row=2, column=1, ipadx=8, padx=5, pady=5, sticky='ew')
        ctk.CTkSlider(self, fg_color=SLIDER_BG, variable=self.rotation, from_=-180, to=180, width=slider_width,
                      command=self.update_rotation).grid(row=3, column=1, ipadx=2, padx=5, pady=5, sticky='ew')
        self.azimuth_value_label = ctk.CTkLabel(self, text=self.azimuth.get(), width=value_width)
        self.rotation_value_label = ctk.CTkLabel(self, text=self.rotation.get(), width=value_width)
        self.azimuth_value_label.grid(row=2, column=3)
        self.rotation_value_label.grid(row=3, column=3)

    def update_azimuth(self, *args):
        value = round(self.azimuth.get(), 1)
        self.azimuth_value_label.configure(text=f'{value}')

    def update_rotation(self, *args):
        value = round(self.rotation.get(), 1)
        self.rotation_value_label.configure(text=f'{value}')


# CALIBRATION ##########################################################################################################

class ChArUcoTarget(ctk.CTkFrame):
    def __init__(self, parent, cali_vars):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.pack(fill='both', pady=4, ipady=8)
        # default dictionary options
        self.default_dictionary = self.master.master.master.master.default_settings['charuco']['dictionary']
        # store variables
        self.cali_vars = cali_vars
        dropdown_width = self.winfo_width()
        # Title Label
        self.title_label = ctk.CTkLabel(self, text="ChArUco Target Configuration", font=("Times", 16))
        self.title_label.pack(fill='x', padx=10, pady=5)

        # Enable/Disable switch
        self.enable_var = ctk.BooleanVar(value=cali_vars['ChArUco'].get())
        self.enable_switch = ctk.CTkSwitch(self, text="Enable ChArUco Target", variable=self.enable_var,
                                           command=self.enable, button_color=BLUE, fg_color=SLIDER_BG)
        self.enable_switch.pack(fill='x', padx=10, pady=5)

        # charuco xy square frame
        self.charuco_xy_frame = ctk.CTkFrame(self, fg_color=DARK_GREY)  # Set desired background color here
        # Dropdown for squares_x
        self.squares_x_var = ctk.IntVar(value=cali_vars['squares_x'].get())
        self.squares_y_var = ctk.IntVar(value=cali_vars['squares_y'].get())

        # Trace changes in dropdown variables
        self.squares_x_var.trace('w', self.update_squares_x)
        self.squares_y_var.trace('w', self.update_squares_y)

        self.label = ctk.CTkLabel(self.charuco_xy_frame, text="Squares (x, y): ")
        self.label.pack(side='left', padx=5, pady=(1, 1))
        self.dropdown_x = ctk.CTkOptionMenu(self.charuco_xy_frame, variable=self.squares_x_var,
                                            values=[str(i) for i in range(5, 40)], width=60)
        self.dropdown_y = ctk.CTkOptionMenu(self.charuco_xy_frame, variable=self.squares_y_var,
                                            values=[str(i) for i in range(5, 40)], width=60)
        self.dropdown_x.pack(side='left', padx=5, pady=(1, 1))
        self.dropdown_y.pack(side='left', padx=5, pady=(1, 1))
        self.charuco_xy_frame.pack(padx=10, pady=(1, 1), fill='x')

        # Add BoxSizePanel
        self.box_size_panel = BoxSizePanel(self, cali_vars, cali_vars['square_length_mm'])
        self.box_size_panel.pack(fill='x', padx=10, pady=(1, 1), expand=False)

        # Dictionary panel options
        default_dictionary = self.master.master.master.master.default_settings['charuco']['dictionary']
        charuco_dict_panel = ChArUcoDictPanel(parent=self, default_dictionary=default_dictionary,
                                              value=cali_vars['dictionary_panel_1'])
        charuco_dict_panel.pack(fill='x', padx=10, pady=(1, 1))

        # detect points switch
        self.detect_var = ctk.BooleanVar(value=cali_vars['detect_corners'].get())
        self.detect_switch = ctk.CTkSwitch(self, text="Detect Corners", variable=self.detect_var,
                                           command=self.detect, button_color=BLUE, fg_color=SLIDER_BG)
        self.detect_switch.pack(fill='x', padx=(20, 10), pady=(1, 1))

        # 3d target enable
        self.target_3d_frame = ctk.CTkFrame(self, fg_color='#3d3d3d')  # Set desired background color here

        self.target_3d_var = ctk.BooleanVar(value=cali_vars['detect_corners'].get())
        self.target_3d_switch = ctk.CTkSwitch(self.target_3d_frame, text="Enable 3D target",
                                              variable=self.target_3d_var,
                                              command=self.target_3d, button_color=BLUE, fg_color=SLIDER_BG)
        self.target_3d_switch.pack(fill='x', padx=(20, 10), pady=(1, 1))

        # Dictionary #2 panel options
        charuco_dict_panel = ChArUcoDictPanel(parent=self.target_3d_frame, default_dictionary=default_dictionary,
                                              value=cali_vars['dictionary_panel_2'], text='Target #2')
        charuco_dict_panel.pack(fill='x', padx=(10, 10), pady=(1, 1))

        # Dictionary #2 panel options
        charuco_dict_panel = ChArUcoDictPanel(parent=self.target_3d_frame, default_dictionary=default_dictionary,
                                              value=cali_vars['dictionary_panel_3'], text='Target #3')
        charuco_dict_panel.pack(fill='x', padx=(10, 10), pady=(1, 5))

        self.target_3d_frame.pack(padx=(5, 5), pady=(5, 5), fill='x')

        # Bind the configure event to check the width
        # self.bind("<Configure>", self.check_width)
        # print(f'ChArUcoTarget width 1: {self.winfo_width()}')

    def update_squares_x(self, *args):
        self.cali_vars['squares_x'].set(self.squares_x_var.get())
        # print('Squares X: ' + str(self.cali_vars['squares_x'].get()))

    def update_squares_y(self, *args):
        self.cali_vars['squares_y'].set(self.squares_y_var.get())
        # print('Squares Y: ' + str(self.cali_vars['squares_y'].get()))

    def target_3d(self, *args):
        detect_corners = self.cali_vars['target_3d']
        detect_corners.set(False) if detect_corners.get() else detect_corners.set(True)
        # print('3D target: ' + str(self.cali_vars['target_3d'].get()))

    def detect(self, *args):
        detect_corners = self.cali_vars['detect_corners']
        detect_corners.set(False) if detect_corners.get() else detect_corners.set(True)
        # print('Detect Corners: ' + str(self.cali_vars['detect_corners'].get()))

    def enable(self, *args):
        self.cali_vars['ChArUco'].set(False) if self.cali_vars['ChArUco'].get() else self.cali_vars['ChArUco'].set(True)
        # print('Charuco Target: ' + str(self.cali_vars['ChArUco'].get()))
        self.master.master.master.master.toggle_image_charuco()

    def update_squares_x(self, *args):
        self.cali_vars['squares_x'].set(self.squares_x_var.get())
        # print('Squares X: ' + str(self.cali_vars['squares_x'].get()))

    def check_width(self, event):
        # print(f'ChArUcoTarget width: {self.winfo_width()}')
        pass


class ChArUcoDictPanel(Panel):
    def __init__(self, parent, default_dictionary, value, text='Target #1'):
        super().__init__(parent=parent)
        self.value, self.text = value, text

        # Configure grid layout
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # Label
        self.label = ctk.CTkLabel(self, text=text)
        self.label.grid(row=0, column=0, padx=10, pady=(1, 1), sticky='w')

        # Dropdown Menu
        self.dropdown = ctk.CTkOptionMenu(self, values=default_dictionary, variable=self.value,
                                          command=self.update_value)
        self.dropdown.grid(row=0, column=1, padx=10, pady=(0, 0), sticky='e')

    def update_value(self, selected_value):
        self.value.set(selected_value)
        # print(self.text + ': ' + selected_value)


class LabeledDropdown(ctk.CTkFrame):
    def __init__(self, parent, label_text, variable, values, dropdown_width=60):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.label = ctk.CTkLabel(self, text=label_text)
        self.label.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky='w')
        self.dropdown = ctk.CTkOptionMenu(self, variable=variable, values=values, width=dropdown_width)
        self.dropdown.grid(row=0, column=2, padx=10, pady=5, sticky='e')

        # Configure grid layout to allocate 3/4 space to label and 1/4 to dropdown
        self.grid_columnconfigure(0, weight=0, uniform="group1")
        self.grid_columnconfigure(1, weight=0, uniform="group1")
        self.grid_columnconfigure(2, weight=0, uniform="group1")


class BoxSizePanel(ctk.CTkFrame):
    def __init__(self, parent, cali_vars, value, label_text='ChArUco box size (mm)'):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.cali_vars = cali_vars

        self.label = ctk.CTkLabel(self, text=label_text)
        self.label.grid(row=0, column=0, padx=10, pady=5, sticky='w')

        self.entry_var = ctk.StringVar(value=str(value.get()))
        self.entry = ctk.CTkEntry(self, textvariable=self.entry_var, width=50)
        self.entry.grid(row=0, column=1, padx=(10, 40), pady=5, sticky='e')

        # Bind the return key event
        self.entry.bind('<Return>', self.update_square_length)

        # Configure grid layout
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)

    def update_square_length(self, event):
        self.cali_vars['square_length_mm'].set(float(self.entry_var.get()))
        # print('square_length_mm = ' + str(self.cali_vars['square_length_mm'].get()))


class GeometricCalibrationPanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color=DARK_GREY)
        self.pack(fill='both', pady=4, ipady=8)

        # Shorthand
        self.camera = self.master.master.master.master.camera
        self.charuco = self.master.master.master.master.charuco
        self.photon = self.master.master.master.master.photon
        self.calibrate = self.master.master.master.master.calibrate
        self.cali_vars = self.master.master.master.master.cali_vars
        self.world = self.master.master.master.master.world
        self.virtual_image = np.array(self.master.master.master.master.image['Virtual Camera'].convert('RGB'))

        # Title Label
        self.title_label = ctk.CTkLabel(self, text="Geometric Calibration", font=("Times", 16))
        self.title_label.pack(fill='x', padx=10, pady=(1, 1))

        # Frame for buttons with background color
        self.button_frame = ctk.CTkFrame(self, fg_color=DARK_GREY)  # Set desired background color here
        self.button_frame.pack(padx=10, pady=(1, 1), fill='x')

        # Button to Open Directory
        self.open_dir_button = ctk.CTkButton(self.button_frame, text='Images Directory', command=self.open_directory,
                                             width=50, height=30, bg_color=DARK_GREY)
        self.open_dir_button.pack(side='left', padx=(15, 5), pady=(1, 1), anchor='center', expand=True)
        # Button to Run Calibration
        self.run_calibration_button = ctk.CTkButton(self.button_frame, text='Calibrate', command=self.run_calibration,
                                                    width=50, height=30, bg_color=DARK_GREY)
        self.run_calibration_button.pack(side='left', padx=(5, 15), pady=(1, 1), anchor='center', expand=True)

        # Check Box for Ground Truth Corners
        self.ground_truth_checkbox = ctk.CTkCheckBox(self, text='Use Ground Truth Corners',
                                                     command=self.update_ground_truth)
        self.ground_truth_checkbox.pack(padx=10, pady=(1, 1))
        # Scrollable Frame for File List
        self.file_list_frame = ctk.CTkScrollableFrame(self, width=180, height=60)  # Fixed height
        self.file_list_frame.pack(padx=10, pady=(1, 1))
        # initialize files list
        self.files = []

    def open_directory(self):
        directory = filedialog.askdirectory(initialdir=os.getcwd())
        if directory:
            self.list_files(directory)

    def list_files(self, directory):
        # initialize files list
        self.files = []
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        valid_extensions = ['.jpg', '.bmp', '.png', '.jpeg', '.tiff']
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    self.files.append(os.path.join(root, file))
        # order files by filename + display to user
        self.files = sorted(self.files)
        for file in self.files:
            file_label = ctk.CTkLabel(self.file_list_frame, text=file, anchor='e')
            file_label.pack(fill='x')

    def run_calibration(self, detected={}, detection_errors={}):
        if not self.files:  # Run Calibration on Virtual Image
            self.calibrate.virtual_camera = True
            # number of panels in virtual image
            panels = ['panel_1', 'panel_2', 'panel_3'] if self.charuco.params['target_3d'].get() else ['panel_1']
            for panel in panels:
                # project board corners
                R1, R2, t = self.world.Rt[:, :3], self.world.aXs_angle, self.world.Rt[:, -1]
                xyz_charuco = np.matmul(R1, self.charuco.board_corners[panel].T) + t[:, np.newaxis]
                self.charuco.projected_board_corners_3d[panel] = np.matmul(R2, xyz_charuco).T
            # get ground truth camera parameters
            self.calibrate.eta_ground_truth = self.camera.get_ground_truth_camera_parameters()
            # generate image without detected points
            # virtual_cam_image = self.photon.to_rgb() if self.cali_vars['detect_corners'].get() else self.virtual_image
            virtual_cam_image = self.photon.to_rgb()
            # calibrate
            self.calibrate.h, self.calibrate.w = self.verify_image_dimensions([virtual_cam_image])
            # detect corners
            for panel in panels:
                detected[panel] = self.charuco.read_chessboard(virtual_cam_image.copy(),
                                                               self.charuco.aruco_dict[panel],
                                                               self.charuco.board[panel])
                # Re-projection errors
                detection_errors[panel] = self.charuco.return_reprojection_errors(detected, panel)
            # if no corners detected show popup and return
            if len(np.concatenate([detection_errors[p] for p in list(detection_errors)])) < 6:
                self.show_popup("No corners detected",
                                "Insufficient corners detected for calibration. Please try again.")
                return
            # record calibration image
            self.calibrate.calibration_images = {'Virtual_Camera': virtual_cam_image}
            # decision: ground truth or detected corners
            self.calibrate.do_camera_calibration(detected_points_2d=[detected],
                                                 initial_positions_3d=self.charuco.board_corners,
                                                 ground_truth_points_2d=[self.charuco.projected_board_corners],
                                                 use_ground_truth_points=self.cali_vars['ground_truth'].get())
        else:
            self.calibrate.virtual_camera = False
            images, images_detected = [cv2.imread(file) for file in self.files], []
            # calibrate
            self.calibrate.h, self.calibrate.w = self.verify_image_dimensions(images)
            # continue
            for image in images:
                # detect corners
                for panel in ['panel_1', 'panel_2', 'panel_3']:
                    detected[panel] = self.charuco.read_chessboard(image,
                                                                   self.charuco.aruco_dict[panel],
                                                                   self.charuco.board[panel])
                images_detected.append(detected)
                detected = {}
            # record images
            _dict_ = {c: [] for c in ['.'.join(f.split(os.sep)[-1].split('.')[:-1]) for f in self.files]}
            self.calibrate.calibration_images = {a: i for a, i in zip(list(_dict_), images)}
            # decision: ground truth or detected corners
            self.calibrate.do_camera_calibration(detected_points_2d=images_detected,
                                                 initial_positions_3d=self.charuco.board_corners)

    def verify_image_dimensions(self, images):
        h, w = images[0].shape[:2]
        for image in images:
            if image.shape[:2] != (h, w):
                self.show_popup("Image Dimensions",
                                "Images in directory are not of the same dimensions. Please try again.")
                return
        return h, w

    @staticmethod
    def show_popup(title, message):
        messagebox.showwarning(title, message)

    # Define the function to update the ground_truth variable
    def update_ground_truth(self):
        self.cali_vars['ground_truth'].set(self.ground_truth_checkbox.get())

    def update_stage_images(self):
        self.cali_vars['stage_images'].set(self.stage_images_checkbox.get())
        # print('Print Stage Images:', self.cali_vars['ground_truth'].get())
