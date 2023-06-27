import tkinter as tk
import cv2
import numpy as np
import pyautogui
import display
import time

def make_window(dimensions):
    (left, top, width, height) = dimensions
    # Create a semi-transparent window
    window = tk.Tk()
    window.attributes('-alpha', 0.5)

    # Set the window size
    window.geometry(str(width) + 'x' + str(height))

    # Position the window at (0,0)
    window.geometry("+" + str(left) + "+" + str(top))

    # Run the main event loop
    window.mainloop()
    
def visualize_points(dimensions, bird_position, hill_points):
    

    (left, top, width, height) = dimensions

    # Create a blank white image
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw the bird position
    if bird_position is not None:
        cv2.circle(image, bird_position, 5, (0, 0, 255), -1)

    # Draw the hill points
    for point in hill_points:
        cv2.circle(image, point, 2, (0, 255, 0), -1)

    # Display the image
    cv2.destroyAllWindows()
    cv2.imshow("Points Visualization", image)