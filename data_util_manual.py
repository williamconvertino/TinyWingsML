import tkinter as tk
from tkinter import filedialog
import os

# Global variables
image_paths = []
current_image_index = 0
labels = []

def open_folder():
    global image_paths, current_image_index
    folder_path = filedialog.askdirectory()
    if folder_path:
        image_paths = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))])
        if image_paths:
            current_image_index = 0
            show_image()

def show_image():
    global image_paths, current_image_index
    if current_image_index < len(image_paths):
        image_path = image_paths[current_image_index]
        img_label.config(text=f"Image {current_image_index+1}/{len(image_paths)}")
        img_label.pack()
        canvas.pack_forget()
        img = tk.PhotoImage(file=image_path)
        canvas.config(width=img.width(), height=img.height())
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img
        canvas.pack()
        entry.focus_set()

def save_label_entry(event=None):
    global labels, current_image_index
    label = entry.get()
    labels.append(label)
    entry.delete(0, tk.END)
    current_image_index += 1
    if current_image_index < len(image_paths):
        show_image()
    else:
        write_labels_to_file()

def write_labels_to_file():
    global labels
    output_file = 'labels.txt'
    with open(output_file, 'w') as f:
        for label in labels:
            f.write(f'{label}\n')
    print(f'Labels saved to {output_file}')

# Create the main window
window = tk.Tk()
window.title("Image Labeling Tool")

# Create canvas to display images
canvas = tk.Canvas(window)

# Create label for image index
img_label = tk.Label(window)

# Create entry field to input labels
entry = tk.Entry(window)
entry.pack()

# Create button to open the image folder
open_folder_button = tk.Button(window, text="Open Folder", command=open_folder)
open_folder_button.pack()

# Create button to submit the label
submit_button = tk.Button(window, text="Submit", command=save_label_entry)
submit_button.pack()

# Bind the Enter key to the submit function
window.bind('<Return>', save_label_entry)

# Start the GUI event loop
window.mainloop()
