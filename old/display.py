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

def visualize_points(image_array, bird_position, hill_points):
    

    # (left, top, width, height) = dimensions

    # Create a blank white image
    # image = np.ones((height, width, 3), dtype=np.uint8) * 255
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Create a white image with the same shape as the screenshot
    white_image = np.ones(image.shape, dtype=np.uint8) * 255

    # Overlay the white image on the screenshot
    image = cv2.addWeighted(white_image, 0.5, image, 0.5, 0)


    # Draw the bird position
    if bird_position is not None:
        cv2.circle(image, bird_position, 5, (0, 0, 255), -1)

    # Draw the hill points
    for point in hill_points:
        cv2.circle(image, point, 2, (0, 255, 0), -1)

    color = (0, 255, 0)  # Green color (BGR)
    thickness = 2  # Thickness of the rectangle border
    x,y,width,height = 600,20,220,30
    cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)


    # Display the image
    cv2.destroyAllWindows()
    cv2.imshow("Points Visualization", image)