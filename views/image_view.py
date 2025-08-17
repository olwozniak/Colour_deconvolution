import tkinter as tk
from PIL import Image, ImageTk


class ImageView(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.original_image = None
        self.current_scale = 1.0
        self.user_zoomed = False
        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self, bg='white')
        self.v_scroll = tk.Scrollbar(self, orient="vertical")
        self.h_scroll = tk.Scrollbar(self, orient="horizontal")
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)
        self.canvas.config(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw")