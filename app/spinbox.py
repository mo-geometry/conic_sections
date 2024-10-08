import customtkinter
from typing import Union, Callable


class WidgetName(customtkinter.CTkFrame):
    def __init__(self, *args,
                 width: int = 100,
                 height: int = 32,
                 **kwargs):
        super().__init__(*args, width=width, height=height, **kwargs)


class FloatSpinbox(customtkinter.CTkFrame):
    def __init__(self, *args,
                 width: int = 100,
                 height: int = 32,
                 step_size: Union[int, float] = 1,
                 min_value: Union[int, float] = 30,
                 max_value: Union[int, float] = 9999999999999,
                 command: Callable = None,
                 z_var=None,  # Add z_var parameter
                 **kwargs):
        super().__init__(*args, width=width, height=height, **kwargs)

        self.step_size = step_size
        self.min_value = min_value  # Set min_value
        self.max_value = max_value  # Set min_value
        self.command = command
        self.z_var = z_var
        self.z_var.trace("w", self.update_spinbox_value)

        self.configure(fg_color=("gray78", "gray28"))  # set frame color

        self.grid_columnconfigure((0, 2), weight=0)  # buttons don't expand
        self.grid_columnconfigure(1, weight=1)  # entry expands

        self.subtract_button = customtkinter.CTkButton(self, text="-", width=height - 6, height=height - 6,
                                                       command=self.subtract_button_callback)
        self.subtract_button.grid(row=0, column=0, padx=(3, 0), pady=3)

        self.entry = customtkinter.CTkEntry(self, width=width - (2 * height), height=height - 6, border_width=0)
        self.entry.grid(row=0, column=1, columnspan=1, padx=3, pady=3, sticky="ew")

        self.add_button = customtkinter.CTkButton(self, text="+", width=height - 6, height=height - 6,
                                                  command=self.add_button_callback)
        self.add_button.grid(row=0, column=2, padx=(0, 3), pady=3)

        # default value
        self.entry.insert(0, str(self.z_var.get()))

    def add_button_callback(self):
        if self.command is not None:
            self.command()
        try:
            value = float(self.entry.get()) + self.step_size
            if value > self.max_value:  # Check if value is less than min_value
                value = self.max_value  # If so, set value to min_value
            self.entry.delete(0, "end")
            self.entry.insert(0, value)
            if self.z_var is not None:  # Check if z_var is set
                self.z_var.set(value)  # Update z_var
        except ValueError:
            return

    def subtract_button_callback(self):
        if self.command is not None:
            self.command()
        try:
            value = float(self.entry.get()) - self.step_size
            if value < self.min_value:  # Check if value is less than min_value
                value = self.min_value  # If so, set value to min_value
            self.entry.delete(0, "end")
            self.entry.insert(0, value)
            if self.z_var is not None:  # Check if z_var is set
                self.z_var.set(value)  # Update z_var
        except ValueError:
            return

    def get(self) -> Union[float, None]:
        try:
            return float(self.entry.get())
        except ValueError:
            return None

    def set(self, value: float):
        self.entry.delete(0, "end")
        self.entry.insert(0, str(float(value)))

    def update_spinbox_value(self, *args):
        self.set(self.z_var.get())
