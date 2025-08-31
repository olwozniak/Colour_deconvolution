import tkinter as tk
from tkinter import messagebox
from PIL import Image
from views.controls import ZoomControls, ColorSpaceControls, DeconvolutionControls
from views.image_view import ImageView
from views.menu import AppMenu
from views.channel_viewer import ChannelViewer
from utils.image_utils import load_image, resize_image
from image_processing.rgb import rgb_split
from image_processing.preprocessing import Preprocessing
import os

class AppController:
    def __init__(self, root):
        self.root = root
        self.root.title("Colour Deconvolution")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.window_width = int(screen_width * 0.4)
        self.window_height = int(screen_height * 0.6)
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.minsize(600, 400)

        self.original_image = None
        self.current_scale = 1.0
        self.user_zoomed = False

        self.image_view = ImageView(self.root, self)
        self.image_view.frame.pack(fill=tk.BOTH, expand=True)

        self.zoom_controls = ZoomControls(self.root, self)
        self.zoom_controls.frame.pack(fill=tk.X, padx=10, pady=5)

        self.color_space_controls = ColorSpaceControls(self.root, self)
        self.color_space_controls.frame.pack(fill=tk.X, padx=10, pady=5)

        self.deconvolution_controls = DeconvolutionControls(self.root, self)
        self.deconvolution_controls.frame.pack_forget()

        self.menu = AppMenu(self.root, self)

        self.apply_button = tk.Button(self.color_space_controls.frame, text="Apply", command=self.apply_color_space_processing)
        self.apply_button.pack(side=tk.RIGHT, padx=5)

        self.root.bind('<Configure>', self.on_window_resize)
        self.preprocessor = Preprocessing()
        self.processed_data = None

    def display_image(self, filepath):
            try:
                self.clear_image()
                self.original_image = load_image(filepath)
                self.image_view.filepath_label.config(text=filepath)
                self.user_zoomed = False
                self.fit_image_to_frame()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")


    def clear_image(self):
        self.image_view.clear_image()
        self.original_image = None
        self.processed_data = None
        self.current_scale = 1.0
        self.user_zoomed = False
        self.zoom_controls.zoom_slider.set(100)

        if hasattr(self, 'hematoxylin_channel'):
            del self.hematoxylin_channel
        if hasattr(self, 'eosin_channel'):
            del self.eosin_channel

    def fit_image_to_frame(self):
        if not self.original_image:
            return

        canvas_width = self.image_view.image_canvas.winfo_width()
        canvas_height = self.image_view.image_canvas.winfo_height()

        width_ratio = canvas_width / self.original_image.width
        height_ratio = canvas_height / self.original_image.height
        self.current_scale = min(width_ratio, height_ratio)

        self.zoom_controls.zoom_slider.set(self.current_scale * 100)
        self.update_image_display()

    def update_image_display(self):
        if not self.original_image:
            return
        self.image_view.update_image_display(self.original_image, self.current_scale)

    def on_window_resize(self, event=None):
        if self.original_image and not self.user_zoomed:
            self.fit_image_to_frame()

    def zoom_image(self, factor):
        self.user_zoomed = True
        self.current_scale *= factor
        self.zoom_controls.zoom_slider.set(self.current_scale * 100)
        self.update_image_display()

    def on_zoom_slider(self, value):
        self.user_zoomed = True
        self.current_scale = float(value) / 100
        self.update_image_display()

    def reset_zoom(self):
        self.user_zoomed = False
        self.fit_image_to_frame()

    def on_color_space_changed(self):
        if self.color_space_controls.cs_vars["Stain Deconvolution"].get():
            self.deconvolution_controls.frame.pack(fill=tk.X, padx=10, pady=5)
        else:
            self.deconvolution_controls.frame.pack_forget()

        selected_spaces = [space for space, var in self.color_space_controls.cs_vars.items() if var.get()]
        print("Selected color spaces:", selected_spaces)

    def apply_color_space_processing(self):
        if not self.original_image:
            messagebox.showerror("Error", "No image uploaded")
            return

        try:
            temp_path = os.path.join(os.path.dirname(__file__), "temp_image.png")
            self.original_image.save(temp_path)
            self.processed_data = self.preprocessor.load_process_data(temp_path)
            os.remove(temp_path)

            if self.color_space_controls.cs_vars["RGB"].get():
                try:
                    temp_path = os.path.join(os.path.dirname(__file__), "temp_image.png")
                    self.original_image.save(temp_path)
                    channel_data = rgb_split(temp_path)
                    ChannelViewer(self.root, "RGB Channels", channel_data)
                    os.remove(temp_path)

                except Exception as e:
                    messagebox.showerror("Error", f"Failed: {str(e)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply color space:\n{str(e)}")

    def method_changed(self, event=None):
        selected = self.deconvolution_controls.selected_method.get()
        print(f"Selected method: {selected}")