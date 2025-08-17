import tkinter as tk
from views.menu import create_menu
from views.image_view import ImageView
from views.controls import ControlPanel


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Colour Deconvolution")
        self.configure_window()
        self.image_view = ImageView(self.root)
        self.control_panel = ControlPanel(self.root, self.image_view)
        create_menu(self.root, self)
        self.image_view.pack(fill=tk.BOTH, expand=True)
        self.control_panel.pack(fill=tk.X)

    def configure_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{int(screen_width * 0.4)}x{int(screen_height * 0.6)}")
        self.root.minsize(600, 400)