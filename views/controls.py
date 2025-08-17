import tkinter as tk
from tkinter import ttk


class ZoomControls:
    def __init__(self, parent, controller):
        self.frame = tk.Frame(parent)
        self.controller = controller

        ttk.Button(self.frame, text="-", width=3, command=lambda: controller.zoom_image(0.8)).pack(side=tk.LEFT, padx=2)
        self.zoom_slider = ttk.Scale(self.frame, from_=10, to=200, command=controller.on_zoom_slider)
        self.zoom_slider.set(100)
        self.zoom_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(self.frame, text="+", width=3, command=lambda: controller.zoom_image(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.frame, text="Reset", command=controller.reset_zoom).pack(side=tk.LEFT, padx=5)


class ColorSpaceControls:
    def __init__(self, parent, controller):
        self.frame = tk.Frame(parent)
        self.controller = controller
        self.cs_vars = {}

        tk.Label(self.frame, text="Colour stain:").pack(side=tk.LEFT, padx=5)

        pb_options = ["RGB", "HSV", "LaB", "Stain Deconvolution"]
        for option in pb_options:
            var = tk.BooleanVar(value=False)
            self.cs_vars[option] = var
            cb = tk.Checkbutton(self.frame, text=option, variable=var, command=controller.on_color_space_changed)
            cb.pack(side=tk.LEFT, padx=5)

        self.cs_vars["RGB"].set(True)


class DeconvolutionControls:
    def __init__(self, parent, controller):
        self.frame = tk.Frame(parent)
        self.controller = controller

        tk.Label(self.frame, text="Deconvolution Method:").pack(side=tk.LEFT, padx=5)
        self.deconv_methods = ["idk", "idk1", "idk2", "idk3"]
        self.selected_method = tk.StringVar()
        self.selected_method.set(self.deconv_methods[0])
        self.method_dropdown = ttk.Combobox(self.frame,
                                            textvariable=self.selected_method,
                                            values=self.deconv_methods,
                                            state="readonly")
        self.method_dropdown.pack(side=tk.LEFT, padx=5)
        self.method_dropdown.bind("<<ComboboxSelected>>", controller.method_changed)