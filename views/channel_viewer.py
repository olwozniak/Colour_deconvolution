import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ChannelViewer:

    def __init__(self, parent, title, channel_data):
        self.top = tk.Toplevel(parent)
        self.top.title(title)

        self.notebook=ttk.Notebook(self.top)

        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.frames = {}
        self.canvases = {}
        self.scrollbars = {}

        for channel_name, channel_image in channel_data.items():

            self.frames[channel_name] = tk.Frame(self.notebook)
            self.notebook.add(self.frames[channel_name], text=channel_name)

            self.scrollbars[channel_name] = {
                'x': ttk.Scrollbar(self.frames[channel_name], orient="horizontal"),
                'y': ttk.Scrollbar(self.frames[channel_name], orient="vertical")
            }

            self.canvases[channel_name] = tk.Canvas(
                self.frames[channel_name],
                xscrollcommand=self.scrollbars[channel_name]['x'].set,
                yscrollcommand=self.scrollbars[channel_name]['y'].set
            )

            self.scrollbars[channel_name]['y'].pack(side=tk.RIGHT, fill=tk.Y)
            self.scrollbars[channel_name]['x'].pack(side=tk.BOTTOM, fill=tk.X)
            self.canvases[channel_name].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            self.scrollbars[channel_name]['x'].config(command=self.canvases[channel_name].xview)
            self.scrollbars[channel_name]['y'].config(command=self.canvases[channel_name].yview)

            self.display_channel(channel_name, channel_image)

    def display_channel(self, channel_name, channel_image):
        tk_image = ImageTk.PhotoImage(channel_image)
        self.canvases[channel_name].image = tk_image
        self.canvases[channel_name].create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.canvases[channel_name].config(scrollregion=(0, 0, channel_image.width, channel_image.height))
