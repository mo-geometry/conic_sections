import customtkinter as ctk
from tkinter import filedialog, Canvas
from app.settings import *
from os import getcwd

# from app.zoom_pan import CanvasImage


class ImageImport(ctk.CTkFrame):
    def __init__(self, parent, import_func):
        super().__init__(master=parent)
        self.grid(column=0, columnspan=2, row=0, sticky='nsew')
        self.import_func = import_func

        ctk.CTkButton(self, text='open image', command=self.open_dialog).pack(expand=True)

    def open_dialog(self):
        try:
            path = filedialog.askopenfile(initialdir=getcwd()).name
            self.import_func(path)
        except FileNotFoundError:
            print("File not found.")
        except AttributeError:
            print("No file selected.")


class ImageOutput(Canvas):
    def __init__(self, parent, resize_image):
        super().__init__(master=parent, background=BACKGROUND_COLOUR, bd=0, highlightthickness=0, relief='ridge')
        self.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        self.bind('<Configure>', resize_image)


class CloseOutput(ctk.CTkButton):
    def __init__(self, parent, close_func):
        super().__init__(master=parent, command=close_func,
                         text='x', text_color=WHITE, fg_color='transparent', width=40, height=40,
                         corner_radius=0, hover_color=CLOSE_RED)
        self.place(relx=0.99, rely=0.01, anchor='ne')


class ImageTabOutput(ctk.CTkTabview):
    def __init__(self, parent, resize_image):
        super().__init__(master=parent)
        self.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        self.parent = parent

        # tabs
        self.add('Virtual Camera')
        self.add('3D World')
        self.add('Image')

        # widgets
        display_tab = TabImage1(self.tab('Image'), resize_image, 'Image')
        analysis_tab = TabImage2(self.tab('3D World'), resize_image, '3D World')
        virtual_camera_tab = TabImage3(self.tab('Virtual Camera'), resize_image, 'Virtual Camera')

        # Configure the grid to expand
        display_tab.grid_columnconfigure(0, weight=1, uniform='a')
        display_tab.grid_rowconfigure(0, weight=1, uniform='a')
        analysis_tab.grid_columnconfigure(0, weight=1, uniform='a')
        analysis_tab.grid_rowconfigure(0, weight=1, uniform='a')
        virtual_camera_tab.grid_columnconfigure(0, weight=1, uniform='a')
        virtual_camera_tab.grid_rowconfigure(0, weight=1, uniform='a')


class TabImage1(Canvas):
    def __init__(self, parent, resize_image, tab_name):
        super().__init__(master=parent, background=BACKGROUND_COLOUR, bd=0, highlightthickness=0, relief='ridge')
        # self.grid(row=0, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.bind('<Configure>', lambda event: resize_image(event, tab_name))


class TabImage2(Canvas):
    def __init__(self, parent, resize_image, tab_name):
        super().__init__(master=parent, background=BACKGROUND_COLOUR, bd=0, highlightthickness=0, relief='ridge')
        # self.grid(row=0, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.bind('<Configure>', lambda event: resize_image(event, tab_name))


class TabImage3(Canvas):
    def __init__(self, parent, resize_image, tab_name):
        super().__init__(master=parent, background=BACKGROUND_COLOUR, bd=0, highlightthickness=0, relief='ridge')
        # self.grid(row=0, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.bind('<Configure>', lambda event: resize_image(event, tab_name))
