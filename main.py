import tkinter as tk
from tkinter import filedialog, Menu
from PIL import Image, ImageTk
from tkinter import messagebox

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Colour Deconvolution")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        window_width = int(screen_width * 0.3)
        window_height = int(screen_height * 0.5)

        self.root.minsize(window_width, window_height)
        self.root.maxsize(screen_width, screen_height)

        self.root.geometry(f"{window_width}x{window_height}")

        self.create_menu()

        self.filepath_label = tk.Label(self.root, text="No file selected", relief=tk.SUNKEN, anchor=tk.W)
        self.filepath_label.pack(fill=tk.X, padx=5, pady=5)

        self.image_frame = tk.Frame(self.root, bg="white")
        self.image_frame.pack(expand=True, fill='both', padx=10, pady=10)

        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack(expand=True)

        self.create_deconvolution_controls()

    def create_menu(self):
        menubar = Menu(self.root)

        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.browse_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        self.root.config(menu=menubar)

    def browse_file(self):
        filetypes = (
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        )

        filename = filedialog.askopenfilename(
            title="Open a file",
            initialdir="/",
            filetypes=filetypes
        )

        if filename:
            self.filepath_label.config(text=filename)
            self.display_image(filename)
            # KOD DO PROCESOWANIA OBRAZU CZY COS JESZCZE NWM

    def display_image(self, filepath):
        try:
            img = Image.open(filepath)

            frame_width = self.image_frame.winfo_width()
            frame_height = self.image_frame.winfo_height()

            # Calculate the ratio to maintain aspect ratio
            img_ratio = img.width / img.height
            frame_ratio = frame_width / frame_height

            # Resize while maintaining aspect ratio
            if frame_ratio > img_ratio:
                new_height = frame_height
                new_width = int(new_height * img_ratio)
            else:
                new_width = frame_width
                new_height = int(new_width / img_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)

            # Update the label
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference

        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

    def create_deconvolution_controls(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        tk.Label(control_frame, text="Deconvolution Method:").pack(side=tk.LEFT, padx=5)

        self.deconv_methods = [
            "idk",
            "idk2",
            "idk3",
        ]

        self.selected_method = tk.StringVar()
        self.selected_method.set(self.deconv_methods[0])  # Set default

        method_dropdown = tk.OptionMenu(
            control_frame,
            self.selected_method,
            *self.deconv_methods,
            command=self.method_changed
        )
        method_dropdown.pack(side=tk.LEFT, padx=5)


    def method_changed(self, event=None):
        """Called when deconvolution method changes"""
        selected = self.selected_method.get()
        print(f"Selected method: {selected}")
        # tu uzupełnic i guess

        # zmiana metod
        if "idk1" in selected:
            self.set_idk_parameters()
        # ... etc ...

    def set_he_parameters(self):
        """Set parameters for idk deconvolution"""
        print("Setting idk parameters")
        #zmiana parametrów


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()