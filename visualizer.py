import tkinter as tk
import cv2
import numpy as np

white_out = True

class Visualizer:
    def __init__(self, window_name, hide_image=False):
        self.window_name = window_name
        self.hide_image = hide_image
        self.canvas = None
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def set_image_array(self, image_array):
        
        image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        self.window_size = (image.shape[1], image.shape[0])
        
        white_image = np.ones(image.shape, dtype=np.uint8) * 255

        if white_out:
            image = cv2.addWeighted(white_image, 0.5, image, 0.5, 0)
        
        if self.hide_image:
            self.canvas = white_image
        else:
            self.canvas = image
            
    def add_score_digit(self, image, x, y, digit_width, digit_height):
        self.canvas[y:y+digit_height, x:x+digit_width] = image

    def set_score(self, score_text, location):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 255, 0)  # BGR color format
        thickness = 2
        cv2.putText(self.canvas, score_text, location, font, font_scale, color, thickness)

    def set_bird(self, bird_coords):
        cv2.circle(self.canvas, bird_coords, 5, (0, 0, 255), -1)

    def set_hills(self, hill_points):
        for point in hill_points:
            cv2.circle(self.canvas, point, 2, (0, 255, 0), -1)

    def show(self):
        cv2.resizeWindow(self.window_name, self.window_size)
        cv2.imshow(self.window_name, self.canvas)