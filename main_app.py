"""
Author: Brian O'Sullivan
Email: bmw.osullivan@gmail.com
Webpage: github.com/mo-geometry
Date: 08-10-2024
"""
import cv2
import platform
import io
import copy
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from PIL import Image, ImageTk, ImageOps, ImageEnhance, ImageFilter
from app.image_widgets import *
from app.menu import *
from app.throttled_var import *

from modules.camera import CAMERA
from modules.world import WORLD
from modules.photon import PHOTON
from modules.charuco import CHARUCO
from modules.calibration import CALIBRATE
from modules.matplotlib_class import *

# VARIABLES ############################################################################################################

APP_WIDTH = 1000
APP_HEIGHT = 777


# MAIN APP #############################################################################################################

class App(ctk.CTk):
    def __init__(self):
        # setup
        super().__init__()
        ctk.set_appearance_mode('dark')
        self.toggle_image_charuco_flag = False
        # Load the .icns file and set it as the window icon
        self.set_icon()

        # Get screen width and height
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = screen_width - APP_WIDTH, 0

        # Set the window geometry
        self.geometry(f'{APP_WIDTH}x{APP_HEIGHT}+{x - 10}+{y + 50}')

        self.geometry('1000x600')
        self.title('MO-GEOMETRY: Virtual Camera')
        self.minsize(1000, 777)

        # layout
        self.rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, minsize=200, weight=0)
        # self.columnconfigure(0, weight=2, uniform='a')
        self.columnconfigure(1, weight=1, uniform='a')

        # canvas data
        self.image_width = {'Image': 0, '3D World': 0}
        self.image_height = {'Image': 0, '3D World': 0}
        self.canvas_width = {'Image': 0, '3D World': 0}
        self.canvas_height = {'Image': 0, '3D World': 0}

        # initialize parameters
        self.init_parameters()

        # widgets
        self.image_import = ImageImport(self, self.import_image)

        # initialize the world object and camera
        self.camera = CAMERA(self)
        self.world = WORLD(self)
        self.photon = PHOTON(self)
        self.plotting = MATPLOTLIB(self)  # camera_obj='bunny.obj', rot_obj_x=-90, rot_obj_y=0, rot_obj_z=0
        self.charuco = CHARUCO(self)
        self.calibrate = CALIBRATE(self)

        # Initialize the position of the image on the canvas
        self.image_position = {'Image': (0, 0), '3D World': (0, 0)}

        # auto open broskow
        # self.import_image(os.path.join(os.getcwd(), 'images', 'broskow.jpeg'))

        # run
        self.mainloop()

    def set_icon(self):
        if platform.system() == 'Darwin':  # macOS
            icon_path = 'app/ico/mo_geometry.icns'
        elif platform.system() == 'Windows':  # Windows
            icon_path = 'app/ico/mo_geometry.ico'
        else:  # Other systems (Linux, etc.)
            icon_path = 'app/ico/mo_geometry.png'  # Assuming a fallback PNG icon
        self.icon_path = icon_path
        icon = Image.open(self.icon_path)
        self.iconphoto(True, ImageTk.PhotoImage(icon))

    def toggle_image_charuco(self, *args):
        if self.toggle_image_charuco_flag:
            return
        self.toggle_image_charuco_flag = True
        try:
            if self.cali_vars['ChArUco'].get():
                # print('Enabling Charuco')
                self.charuco.generate_board()
                # self.charuco.read_chessboards([self.charuco.charuco_board])
                # project corners through model
                self.camera.project_corners_through_model()
                # write detections and show errors
                # self.charuco.write_detected_to_images([self.charuco.charuco_board])
                # toggle from image to charuco
                self.user_loaded_image = self.opencv_image.copy()
                # replace with charuco image
                h, w = self.charuco.charuco_board['panel_1'].shape
                self.opencv_image = np.repeat(self.charuco.charuco_board['panel_1'].reshape(h, w, 1), 3, axis=2)
                # update the image attributes
                self.photon.load_object_image(self.opencv_image, charuco=True)
                # 3d target
                # if np.logical_and(self.charuco.params['target_3d'].get(), self.cali_vars['ChArUco'].get()):
                self.photon.load_object_image(self.charuco.charuco_board['panel_2'], charuco=True, panel='panel_2')
                self.photon.load_object_image(self.charuco.charuco_board['panel_3'], charuco=True, panel='panel_3')
                # replace image
                self.place_image('Virtual Camera')
            elif self.cali_vars['ChArUco'].get() is False:
                # print('Enabling Image')
                # toggle from image to charuco
                self.opencv_image = self.user_loaded_image.copy()
                # update the image attributes
                self.photon.load_object_image(self.opencv_image)
                # replace image
                self.place_image('Virtual Camera')
        finally:
            self.toggle_image_charuco_flag = False

    def init_parameters(self):
        # Load settings from the YAML file
        with open(os.path.join('app', 'settings.yaml'), 'r') as file:
            settings = yaml.safe_load(file)
            self.default_settings = copy.deepcopy(settings)
        self.pos_vars = {'mirror_flip': ctk.StringVar(value=settings['mirror_flip_options'][0])}
        # extrinsics variables
        self.ext_vars = {'x': ctk.StringVar(value=settings['position']['x']),
                         'y': ctk.StringVar(value=settings['position']['y']),
                         'z': ctk.StringVar(value=settings['position']['z']),
                         'roll': ThrottledVar(ctk.DoubleVar(value=settings['euler']['roll']), 0.1),
                         'pitch': ThrottledVar(ctk.DoubleVar(value=settings['euler']['pitch']), 0.1),
                         'yaw': ThrottledVar(ctk.DoubleVar(value=settings['euler']['yaw']), 0.1),
                         'field_angle': ctk.DoubleVar(value=settings['axis-angle']['field_angle']),
                         'azimuth': ThrottledVar(ctk.DoubleVar(value=settings['axis-angle']['azimuth']), 0.1),
                         'rotation': ThrottledVar(ctk.DoubleVar(value=settings['axis-angle']['rotation']), 0.1),
                         'rpy_convention': ctk.StringVar(value=settings['rpy_convention'][0])
                         }
        # intrinsics variables
        self.int_vars = {'focal_length': ctk.DoubleVar(value=settings['camera_matrix']['focal_length']),
                         'dX': ctk.StringVar(value=settings['camera_matrix']['dX']),
                         'dY': ctk.StringVar(value=settings['camera_matrix']['dY']),
                         'yx_aspect_ratio': ctk.StringVar(value=settings['camera_matrix']['yx_aspect_ratio']),
                         'lens_distortion': ctk.StringVar(value=settings['lens_distortion'][-1]),
                         'width': ctk.DoubleVar(value=settings['sensor']['width']),
                         'height': ctk.DoubleVar(value=settings['sensor']['height']),
                         'tilt_angle': ThrottledVar(ctk.DoubleVar(value=settings['sensor']['tilt_angle']), 0.1),
                         'tilt_azimuth': ThrottledVar(ctk.DoubleVar(value=settings['sensor']['tilt_azimuth']), 0.1),
                         're-projection_type': ctk.StringVar(value=settings['re-projection']['type'][0]),
                         're-projection_mode': ctk.StringVar(value=settings['re-projection']['mode'][0])}
        # calibration variables
        self.cali_vars = {'ChArUco': ctk.BooleanVar(value=False),
                          'squares_y': ctk.IntVar(value=settings['charuco']['squares_y']),
                          'squares_x': ctk.IntVar(value=settings['charuco']['squares_x']),
                          'detect_corners': ctk.BooleanVar(value=False),
                          'target_3d': ctk.BooleanVar(value=settings['charuco']['target_3d']),
                          'ground_truth': ctk.BooleanVar(value=settings['charuco']['ground_truth']),
                          'square_length_mm': ctk.DoubleVar(value=settings['charuco']['square_length_mm']),
                          'dictionary_panel_1': ctk.StringVar(value=settings['charuco']['dictionary'][3]),
                          'dictionary_panel_2': ctk.StringVar(value=settings['charuco']['dictionary'][2]),
                          'dictionary_panel_3': ctk.StringVar(value=settings['charuco']['dictionary'][1]),
                          'stage_images': ctk.BooleanVar(value=settings['stage_images']),
                          'use_image_center': ctk.BooleanVar(value=settings['use_image_center']),
                          }
        # tracing
        for var in list(self.pos_vars.values()):
            var.trace('w', self.manipulate_image)

        for var in [self.cali_vars['detect_corners'], self.cali_vars['target_3d']]:
            var.trace('w', self.manipulate_virtual_camera)

        # camera-world variables
        camera_world_vars = list(self.ext_vars.values()) + list(self.int_vars.values())
        for var in camera_world_vars:
            if isinstance(var, ThrottledVar):
                var.var.trace('w', self.manipulate_virtual_camera)
            else:
                var.trace('w', self.manipulate_virtual_camera)
        self.prev_int_vars = {var_name: var.get() for var_name, var in self.int_vars.items()}
        self.prev_ext_vars = {var_name: var.get() for var_name, var in self.ext_vars.items()}

    def manipulate_virtual_camera(self, *args, int_vars_changed=False, ext_vars_changed=False):
        current_tab = self.image_output.get()
        for var_name, var in self.int_vars.items():
            if self.prev_int_vars[var_name] != var.get():
                # print(f"The value of {var_name} in int_vars has changed to {var.get()}")
                int_vars_changed = True
                self.camera.initialize_intrinsics()
                # intrinsics change update background
                if np.logical_or(var_name != 're-projection_type', var_name != 're-projection_mode'):
                    self.photon.initialize_background()
                if current_tab == '3D World':
                    # self.plotting.update_intrinsics_figure()
                    pass
                if self.plotting.lens_grid_figure_open:
                    self.plotting.update_lens_grid_figure()

        for var_name, var in self.ext_vars.items():
            if self.prev_ext_vars[var_name] != var.get():
                # print(f"The value of {var_name} in ext_vars has changed to {var.get()}")
                ext_vars_changed = True
                self.world.initialize_extrinsics()
                if current_tab == '3D World':
                    self.plotting.update_figure()
        if int_vars_changed:
            self.prev_int_vars = {var_name: var.get() for var_name, var in self.int_vars.items()}
        if ext_vars_changed:
            self.prev_ext_vars = {var_name: var.get() for var_name, var in self.ext_vars.items()}
        if current_tab == '3D World':
            self.prev_ext_vars = {var_name: var.get() for var_name, var in self.ext_vars.items()}
            self.prev_int_vars = {var_name: var.get() for var_name, var in self.int_vars.items()}

        self.place_image(current_tab)

    def manipulate_image(self, *args):
        current_tab = self.image_output.get()
        self.image['Image'] = self.original
        # mirror flip
        if self.pos_vars['mirror_flip'].get() != MIRROR_FLIP_OPTIONS[0]:
            if self.pos_vars['mirror_flip'].get() == 'mirror':
                self.image['Image'] = ImageOps.mirror(self.image['Image'])
            if self.pos_vars['mirror_flip'].get() == 'flip':
                self.image['Image'] = ImageOps.flip(self.image['Image'])
            if self.pos_vars['mirror_flip'].get() == 'mirror&flip':
                self.image['Image'] = ImageOps.mirror(ImageOps.flip(self.image['Image']))

        # update the display image
        self.place_image(current_tab)

    def virtual_camera(self, target_image_rgb):
        self.photon.load_object_image(target_image_rgb)
        camera_image_rgb = self.photon.to_rgb()
        return camera_image_rgb

    def import_image(self, path):
        self.opencv_image = cv2.imread(path)[:, :, ::-1]
        if max(self.opencv_image.shape) > 10 ** 3:
            h, w = self.opencv_image.shape[0], self.opencv_image.shape[1]
            m = np.round(max(h, w) / 960, 3)
            self.opencv_image = cv2.resize(self.opencv_image, (int(w / m), int(h / m)),
                                           interpolation=cv2.INTER_LINEAR)
        self.original = Image.fromarray(self.opencv_image)
        self.image = {'Image': self.original.copy(),
                      '3D World': Image.open(os.path.join('images', 'broskow.jpeg')),
                      'Virtual Camera': Image.fromarray(self.virtual_camera(self.opencv_image))}
        self.image_ratio = {'Image': self.image['Image'].size[0] / self.image['Image'].size[1],
                            '3D World': self.image['3D World'].size[0] / self.image['3D World'].size[1],
                            'Virtual Camera':
                                self.default_settings['sensor']['width'] / self.default_settings['sensor']['height']}
        self.image_tk = {'Image': ImageTk.PhotoImage(self.image['Image']),
                         '3D World': ImageTk.PhotoImage(self.image['3D World']),
                         'Virtual Camera': ImageTk.PhotoImage(self.image['Virtual Camera'])}
        self.image_import.grid_forget()
        self.image_output = ImageTabOutput(self, self.resize_image)
        self.close_button = CloseOutput(self, self.close_edit)
        self.menu = Menu(self, self.pos_vars, self.ext_vars, self.int_vars, self.cali_vars, self.export_image)
        # initialize extrinsics plot
        self.plotting.update_figure()

    def close_edit(self):
        # hide the image and the close button
        self.image_output.grid_forget()
        self.image_output.place_forget()
        self.menu.grid_forget()
        # reinitialize the variable
        self.revert_parameters()
        # recreate the import button
        self.image_import = ImageImport(self, self.import_image)

    def revert_parameters(self):
        # intrinsic parameters
        params_int = (
            (self.int_vars['focal_length'], self.default_settings['camera_matrix']['focal_length']),
            (self.int_vars['dX'], self.default_settings['camera_matrix']['dX']),
            (self.int_vars['dY'], self.default_settings['camera_matrix']['dY']),
            (self.int_vars['yx_aspect_ratio'], self.default_settings['camera_matrix']['yx_aspect_ratio']),
            (self.int_vars['lens_distortion'], self.default_settings['lens_distortion'][-1]),
            (self.int_vars['tilt_angle'], self.default_settings['sensor']['tilt_angle']),
            (self.int_vars['tilt_azimuth'], self.default_settings['sensor']['tilt_azimuth']),
            (self.int_vars['re-projection_type'], self.default_settings['re-projection']['type'][0]),
            (self.int_vars['re-projection_mode'], self.default_settings['re-projection']['mode'][0])
        )
        self.revert_tuple_list(params_int)
        # extrinsic parameters
        params_ext = (
            (self.pos_vars['mirror_flip'], MIRROR_FLIP_OPTIONS[0]),
            (self.ext_vars['x'], self.default_settings['position']['x']),
            (self.ext_vars['y'], self.default_settings['position']['y']),
            (self.ext_vars['z'], self.default_settings['position']['z']),
            (self.ext_vars['roll'], self.default_settings['euler']['roll']),
            (self.ext_vars['pitch'], self.default_settings['euler']['pitch']),
            (self.ext_vars['yaw'], self.default_settings['euler']['yaw']),
            (self.ext_vars['field_angle'], self.default_settings['axis-angle']['field_angle']),
            (self.ext_vars['azimuth'], self.default_settings['axis-angle']['azimuth']),
            (self.ext_vars['rotation'], self.default_settings['axis-angle']['rotation']),
            (self.ext_vars['rpy_convention'], self.default_settings['rpy_convention'][0])
        )
        self.revert_tuple_list(params_ext)

    @staticmethod
    def revert_tuple_list(tuple_list):
        for var, value in tuple_list:
            if isinstance(var, ThrottledVar):
                var.var.set(value)
            else:
                var.set(value)

    def resize_image(self, event, tab_name):
        # Get the Canvas widget for the current tab
        canvas_tab = self.image_output.tab(tab_name)
        self.canvas_width[tab_name] = canvas_tab.winfo_width()
        self.canvas_height[tab_name] = canvas_tab.winfo_height()

        # resize
        canvas_ratio = self.canvas_width[tab_name] / self.canvas_height[tab_name]
        if canvas_ratio > self.image_ratio[tab_name]:  # image is wider than the image
            self.image_height[tab_name] = int(event.height)
            self.image_width[tab_name] = int(self.image_height[tab_name] * self.image_ratio[tab_name])
        else:  # canvas is taller than the image
            self.image_width[tab_name] = int(event.width)
            self.image_height[tab_name] = int(self.image_width[tab_name] / self.image_ratio[tab_name])
        # call place function
        self.place_image(tab_name)

    def print_viewpoint(self, event):
        ax = self.plotting.figure['fig'].axes[0]
        elev, azim, roll = np.round(ax.elev, 3), np.round(ax.azim, 3), np.round(ax.roll, 3)
        # print(f"Elevation: {elev}, Azimuth: {azim}, Roll: {roll}")

    def place_image(self, tab_name):
        # print('current tab is: ', self.image_output.get())
        # place image
        # Get the Canvas widget for the current tab
        canvas_tab = self.image_output.tab(tab_name)
        canvas = canvas_tab.winfo_children()[0]  # Assuming the Canvas is the first child widget

        self.canvas_width[tab_name] = canvas_tab.winfo_width()
        self.canvas_height[tab_name] = canvas_tab.winfo_height()

        # If the tab is '3D World', create a matplotlib figure and add it to the canvas
        if tab_name == '3D World':
            # Delete the current images on the Canvas widgets
            canvas.delete('all')

            # Redraw the figure
            self.plotting.figure['fig'].canvas.draw()

            # Use plt.pause() to allow the figure to update
            plt.pause(0.001)

            # Flush events to process any pending GUI events
            self.plotting.figure['fig'].canvas.flush_events()

            # # Connect the callback to the draw event
            # self.plotting.figure['fig'].canvas.mpl_connect('draw_event', self.print_viewpoint)

            # Check if the canvas already exists
            if not hasattr(self, 'figure_canvas') or self.figure_canvas is None:
                # Create a FigureCanvasTkAgg object with the figure
                self.figure_canvas = FigureCanvasTkAgg(self.plotting.figure['fig'],
                                                       master=self.image_output.tab(tab_name))
                self.figure_canvas.draw()  # Ensure the figure is drawn
                self.figure_canvas.get_tk_widget().pack()
            else:
                # Just redraw the existing canvas
                self.figure_canvas.draw()
        elif tab_name == 'Virtual Camera':
            virtual_cam_image = self.photon.to_rgb()
            if np.logical_and(self.cali_vars['detect_corners'].get(), self.cali_vars['ChArUco'].get()):
                # project corners through model
                self.camera.project_corners_through_model()
                # write detections and show errors
                virtual_cam_image = self.charuco.update_corners_virtual_camera(virtual_cam_image)
            self.image[tab_name] = Image.fromarray(virtual_cam_image)
            self.image_tk[tab_name] = ImageTk.PhotoImage(self.image[tab_name])

        # Resize the image
        try:
            resized_image = self.image[tab_name].resize((self.image_width[tab_name], self.image_height[tab_name]))
            self.image_tk[tab_name] = ImageTk.PhotoImage(resized_image)
        except (AttributeError, KeyError, TypeError) as e:
            print(f"An error occurred: {e} tab is selected. "
                  f"Updates will not be applied here.")

        # Display the image on the Canvas widgets
        if tab_name != '3D World':
            canvas.create_image(self.canvas_width[tab_name] / 2, self.canvas_height[tab_name] / 2,
                                image=self.image_tk[tab_name])

    def export_image(self, name, file, path):
        export_string = f'{path}/{name}.{file}'
        if self.image_output.get() == '3D World':
            img, _ = self.get_image_from_figure(self.plotting.figure['fig'])
            img.save(export_string)
        else:
            self.image[self.image_output.get()].save(export_string)

    @staticmethod
    def get_image_from_figure(fig):
        # Save the figure to a StringIO object
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches='tight', format='png')
        buf.seek(0)

        # Load the StringIO object into a PIL image
        img = Image.open(buf)

        # Create an ImageTk.PhotoImage object
        img_tk = ImageTk.PhotoImage(img)

        # Close the buffer
        buf.close()

        return img, img_tk


App()
