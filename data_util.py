import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import os

# Global variables
image_paths = []
current_image_index = 0
coordinates = []

def open_folder():
    global image_paths, current_image_index
    folder_path = filedialog.askdirectory()
    if folder_path:
        image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
        if image_paths:
            current_image_index = 0
            show_image()

def show_image():
    global image_paths, current_image_index
    if current_image_index < len(image_paths):
        image_path = image_paths[current_image_index]
        image = Image.open(image_path)
        img_tk = ImageTk.PhotoImage(image)
        canvas.configure(width=image.width, height=image.height)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk
        canvas.bind('<Button 1>', capture_coordinates)

def capture_coordinates(event):
    global coordinates, current_image_index
    x = event.x
    y = event.y
    coordinates.append(f'{x}, {y}')
    entry.delete(0, tk.END)
    current_image_index += 1
    if current_image_index < len(image_paths):
        show_image()
    else:
        write_coordinates_to_file()

def save_coordinates():
    global image_paths, current_image_index, coordinates
    entry_value = entry.get()
    if entry_value:
        coordinates.append(entry_value)
    entry.delete(0, tk.END)
    current_image_index += 1
    if current_image_index < len(image_paths):
        show_image()
    else:
        write_coordinates_to_file()

def write_coordinates_to_file():
    global image_paths, coordinates
    output_file = 'positions.txt'
    with open(output_file, 'w') as f:
        for i, image_path in enumerate(image_paths):
            f.write(f'{coordinates[i]}\n')
    print(f'Coordinates saved to {output_file}')

# Create the main window
window = tk.Tk()
window.title("Object Labeling Tool")

# Create canvas to display images
canvas = tk.Canvas(window)
canvas.pack()

# Create entry field to input coordinates
entry = tk.Entry(window)
entry.pack()

# Create button to open the image folder
open_folder_button = tk.Button(window, text="Open Folder", command=open_folder)
open_folder_button.pack()

# Create button to save coordinates
save_button = tk.Button(window, text="Save Coordinates", command=save_coordinates)
save_button.pack()

# Start the GUI event loop
window.mainloop()
