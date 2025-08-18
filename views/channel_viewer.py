import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class ChannelViewer:

    def __init__(self, parent, title, channel_data, temp_files=None):
        self.top = tk.Toplevel(parent)
        self.top.title(title)

        screen_width = self.top.winfo_screenwidth()
        screen_height = self.top.winfo_screenheight()
        self.window_width = int(screen_width *0.5)
        self.window_height = int(screen_height *0.6)
        self.top.geometry(f"{self.window_width}x{self.window_height}")
        self.top.minsize(600, 400)

        self.top.protocol("WM_DELETE_WINDOW", self.on_close)
        self.temp_files = temp_files or []

        self.notebook=ttk.Notebook(self.top)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.channel_data = channel_data
        self.zoom_level = 1.0
        self.tk_images = {}

        for name, image in channel_data.items():
            frame = tk.Frame(self.notebook)
            self.notebook.add(frame, text=name)

            v_scroll = ttk.Scrollbar(frame, orient="vertical")
            h_scroll = ttk.Scrollbar(frame, orient="horizontal")
            v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

            canvas = tk.Canvas(frame, yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            v_scroll.config(command=canvas.yview)
            h_scroll.config(command=canvas.xview)

            setattr(self, f"{name.lower().replace(' ', '_')}_canvas", canvas)
            self.display_image(canvas, image)

        self.zoom_frame = tk.Frame(self.top)
        self.zoom_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(self.zoom_frame, text="-", width=3,
                   command=lambda: self.zoom_all(0.8)).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.zoom_frame, text="+", width=3,
                   command=lambda: self.zoom_all(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.zoom_frame, text="Reset",
                   command=self.reset_zoom).pack(side=tk.LEFT, padx=5)

        ttk.Button(self.zoom_frame, text="Save All",
                   command=self.save_current_channels).pack(side=tk.RIGHT, padx=5)

    def display_image(self, canvas, image):
        width = int(image.width * self.zoom_level)
        height = int(image.height * self.zoom_level)

        img = image.resize((width, height), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(img)

        canvas.tk_image = tk_image
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.config(scrollregion=(0, 0, width, height))

    def zoom_all(self, factor):
        self.zoom_level *= factor
        for name, img in self.channel_data.items():
            canvas = getattr(self, f"{name.lower().replace(' ', '_')}_canvas")
            self.display_image(canvas, img)

    def reset_zoom(self):
        self.zoom_level = 1.0
        for name, img in self.channel_data.items():
            canvas = getattr(self, f"{name.lower().replace(' ', '_')}_canvas")
            self.display_image(canvas, img)

    def save_current_channels(self):
        if not self.channel_data:
            return

        save_dir = filedialog.askdirectory(title="Select directory to save current channels")
        if not save_dir:
            return

        try:
            for name, image in self.channel_data.items():
                clean_name = ''.join(c if c.isalnum() else '_' for c in name)
                save_path = os.path.join(save_dir, f"{clean_name}.png")
                image.save(save_path)

            messagebox.showinfo("Success", f"Current window channels saved to:\n{save_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save current window files:\n{str(e)}")

    def on_close(self):
        for file in self.temp_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except:
                pass
        self.top.destroy()