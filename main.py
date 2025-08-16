import tkinter as tk
from tkinter import filedialog, Menu, messagebox, ttk
from PIL import Image, ImageTk

class App:
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

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.create_widgets()
        self.create_menu()
        self.create_zoom_controls()
        self.create_deconvolution_controls()

        self.root.bind('<Configure>', self.on_window_resize)

    def create_widgets(self):
        self.filepath_label = tk.Label(self.main_frame, text="No file selected",
                                       relief=tk.SUNKEN, anchor=tk.W)
        self.filepath_label.pack(fill=tk.X, padx=5, pady=5)

        self.image_container = tk.Frame(self.main_frame)
        self.image_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.v_scroll = ttk.Scrollbar(self.image_container, orient="vertical")
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.h_scroll = ttk.Scrollbar(self.image_container, orient="horizontal")
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas - packed last to fill remaining space
        self.image_canvas = tk.Canvas(self.image_container,
                                      xscrollcommand=self.h_scroll.set,
                                      yscrollcommand=self.v_scroll.set,
                                      bg='white')
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.h_scroll.config(command=self.image_canvas.xview)
        self.v_scroll.config(command=self.image_canvas.yview)

        self.image_on_canvas = self.image_canvas.create_image(0, 0, anchor="nw")

    def toggle_scrollbars(self):
        if not self.original_image:
            return

        self.image_canvas.pack_forget()
        self.h_scroll.pack_forget()
        self.v_scroll.pack_forget()

        img_width = self.original_image.width * self.current_scale
        img_height = self.original_image.height * self.current_scale
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        need_hscroll = img_width > canvas_width
        need_vscroll = img_height > canvas_height

        # Repack in correct order
        if need_vscroll:
            self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        if need_hscroll:
            hscroll_frame = tk.Frame(self.image_container)
            hscroll_frame.pack(side=tk.BOTTOM, fill=tk.X)
            self.h_scroll = ttk.Scrollbar(hscroll_frame, orient="horizontal")
            self.h_scroll.pack(fill=tk.X)
            self.h_scroll.config(command=self.image_canvas.xview)
            self.image_canvas.config(xscrollcommand=self.h_scroll.set)

        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def update_image_display(self):
        if not self.original_image:
            return

        width = int(self.original_image.width * self.current_scale)
        height = int(self.original_image.height * self.current_scale)
        img = self.original_image.resize((width, height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_canvas.itemconfig(self.image_on_canvas, image=self.tk_image)
        self.image_canvas.config(scrollregion=(0, 0, width, height))

    def on_window_resize(self, event=None):
        if self.original_image and not self.user_zoomed:
            self.fit_image_to_frame()

    def fit_image_to_frame(self):
        if not self.original_image:
            return

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        width_ratio = canvas_width / self.original_image.width
        height_ratio = canvas_height / self.original_image.height
        self.current_scale = min(width_ratio, height_ratio)

        self.zoom_slider.set(self.current_scale * 100)
        self.update_image_display()
        self.toggle_scrollbars()

    def zoom_image(self, factor):
        self.user_zoomed = True
        self.current_scale *= factor
        self.zoom_slider.set(self.current_scale * 100)
        self.update_image_display()
        self.toggle_scrollbars()

    def on_zoom_slider(self, value):
        self.user_zoomed = True
        self.current_scale = float(value) / 100
        self.update_image_display()
        self.toggle_scrollbars()

    def reset_zoom(self):
        self.user_zoomed = False
        self.fit_image_to_frame()

    def display_image(self, filepath):
        try:
            self.original_image = Image.open(filepath)
            self.user_zoomed = False
            self.fit_image_to_frame()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

    def create_menu(self):
        menubar = Menu(self.root)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.browse_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def browse_file(self):
        filetypes = (("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        filename = filedialog.askopenfilename(title="Open a file", initialdir="/", filetypes=filetypes)
        if filename:
            self.filepath_label.config(text=filename)
            self.display_image(filename)

    def create_zoom_controls(self):
        zoom_frame = tk.Frame(self.main_frame)
        zoom_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(zoom_frame, text="-", width=3, command=lambda: self.zoom_image(0.8)).pack(side=tk.LEFT, padx=2)
        self.zoom_slider = ttk.Scale(zoom_frame, from_=10, to=200, command=self.on_zoom_slider)
        self.zoom_slider.set(100)
        self.zoom_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(zoom_frame, text="+", width=3, command=lambda: self.zoom_image(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Reset", command=self.reset_zoom).pack(side=tk.LEFT, padx=5)

    def create_deconvolution_controls(self):
        control_frame = tk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(control_frame, text="Deconvolution Method:").pack(side=tk.LEFT, padx=5)
        self.deconv_methods = ["idk", "idk2", "idk3"]
        self.selected_method = tk.StringVar()
        self.selected_method.set(self.deconv_methods[0])
        method_dropdown = ttk.Combobox(control_frame, textvariable=self.selected_method,
                                       values=self.deconv_methods, state="readonly")
        method_dropdown.pack(side=tk.LEFT, padx=5)
        method_dropdown.bind("<<ComboboxSelected>>", self.method_changed)

    def method_changed(self, event=None):
        selected = self.selected_method.get()
        print(f"Selected method: {selected}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()