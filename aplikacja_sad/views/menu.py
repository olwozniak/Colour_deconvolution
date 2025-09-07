import tkinter as tk
from tkinter import Menu, filedialog, messagebox


class AppMenu:
    def __init__(self, root, controller):
        self.controller = controller
        self.menubar = Menu(root)

        file_menu = Menu(self.menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.browse_file)
        file_menu.add_command(label="Clear", command=controller.clear_image)
        self.menubar.add_cascade(label="File", menu=file_menu)

        root.config(menu=self.menubar)

    def browse_file(self):
        filetypes = (("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        filename = filedialog.askopenfilename(title="Open a file", initialdir="/", filetypes=filetypes)
        if filename:
            self.controller.display_image(filename)