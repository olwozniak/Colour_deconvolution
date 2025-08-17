import tkinter as tk
from tkinter import ttk


class ControlPanel(tk.Frame):
    def __init__(self, parent, image_view):
        super().__init__(parent)
        self.image_view = image_view
        self.create_zoom_controls()
        self.create_color_space_controls()
        self.create_deconvolution_controls()

    def create_zoom_controls(self):
        zoom_frame = tk.Frame(self)
        zoom_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(zoom_frame, text="-", width=3, command=lambda: self.image_view.zoom_image(0.8)).pack(side=tk.LEFT,
                                                                                                        padx=2)
        self.zoom_slider = ttk.Scale(zoom_frame, from_=10, to=200, command=self.image_view.on_zoom_slider)
        self.zoom_slider.set(100)
        self.zoom_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(zoom_frame, text="+", width=3, command=lambda: self.image_view.zoom_image(1.2)).pack(side=tk.LEFT,
                                                                                                        padx=2)
        ttk.Button(zoom_frame, text="Reset", command=self.image_view.reset_zoom).pack(side=tk.LEFT, padx=5)