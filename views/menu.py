from tkinter import Menu

def create_menu(root, app):
    menubar = Menu(root)
    file_menu = Menu(menubar, tearoff=0)
    file_menu.add_command(label="Open", command=app.browse_file)
    file_menu.add_command(label="Clear", command=app.clear_image)
    menubar.add_cascade(label="File", menu=file_menu)
    root.config(menu=menubar)