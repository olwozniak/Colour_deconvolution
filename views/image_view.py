import tkinter as tk
from tkinter import ttk
from PIL import ImageTk


class ImageView:
    def __init__(self, parent, controller):
        self.controller = controller
        self.frame = tk.Frame(parent)

        self.filepath_label = tk.Label(self.frame, text="No file selected",
                                       relief=tk.SUNKEN, anchor=tk.W)
        self.filepath_label.pack(fill=tk.X, padx=5, pady=5)

        self.image_container = tk.Frame(self.frame)
        self.image_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.v_scroll = ttk.Scrollbar(self.image_container, orient="vertical")
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.h_scroll = ttk.Scrollbar(self.image_container, orient="horizontal")
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.image_canvas = tk.Canvas(self.image_container,
                                      xscrollcommand=self.h_scroll.set,
                                      yscrollcommand=self.v_scroll.set,
                                      bg='white')
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.h_scroll.config(command=self.image_canvas.xview)
        self.v_scroll.config(command=self.image_canvas.yview)

        self.image_on_canvas = self.image_canvas.create_image(0, 0, anchor="nw")
        self.tk_image = None

    def update_image_display(self, image, scale):
        width = int(image.width * scale)
        height = int(image.height * scale)
        img = image.resize((width, height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_canvas.itemconfig(self.image_on_canvas, image=self.tk_image)
        self.image_canvas.config(scrollregion=(0, 0, width, height))

    def clear_image(self):
        self.image_canvas.delete("all")
        self.image_on_canvas = self.image_canvas.create_image(0, 0, anchor="nw")
        self.tk_image = None
        self.filepath_label.config(text="No file selected")
        self.image_canvas.config(scrollregion=(0, 0, 1, 1))