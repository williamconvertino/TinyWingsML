import tkinter as tk
import cv2
import numpy as np

white_out = True

class Visualizer:
    def __init__(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def set_image_array(self, image_array, hide_image=False):
        
        image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        self.window_size = (image.shape[1], image.shape[0])
        
        white_image = np.ones(image.shape, dtype=np.uint8) * 255

        if white_out:
            image = cv2.addWeighted(white_image, 0.5, image, 0.5, 0)
        
        if hide_image:
            self.canvas = white_image
        else:
            self.canvas = image
            
    def set_bird(self, bird_coords):
        cv2.circle(self.canvas, bird_coords, 5, (0, 0, 255), -1)

    def set_hills(self, hill_points):
        for point in hill_points:
            cv2.circle(self.canvas, point, 2, (0, 255, 0), -1)

    def show(self):
        cv2.resizeWindow(self.window_name, self.window_size)
        cv2.imshow(self.window_name, self.canvas)