from bird_tracker import BirdTracker
from visualizer import Visualizer
import cv2
import numpy as np

hill_threshold = 220

class EnvironmentReader:
    def __init__(self):
        self.bird_tracker = BirdTracker()    
        self.visualizer = None

        self.screenshot_array = None
        self.hill_points = None
        self.bird_point = None

    def read(self, screenshot):
        self.screenshot_array = np.array(screenshot)

        self.update_hill_points(self.screenshot_array)
        self.update_bird_point(screenshot)

    def update_hill_points(self, screenshot_array):

        height, width, _ = screenshot_array.shape

        hill_points = [(i, height) for i in range(width)]

        grayscale_image = cv2.cvtColor(screenshot_array, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(grayscale_image, hill_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            for point in contour:
                x, y = point[0]
                if y < hill_points[x][1]: 
                    hill_points[x] = (x, y)

        self.hill_points = hill_points

    def update_bird_point(self, screenshot):
        self.bird_coords = self.bird_tracker.get_coords(screenshot)

    def visualize_window(self, hide_image=False):
        if self.visualizer == None:
            self.visualizer = Visualizer("Bird Visualization")

        self.visualizer.set_image_array(self.screenshot_array, hide_image=hide_image)
        self.visualizer.set_bird(self.bird_coords)
        self.visualizer.set_hills(self.hill_points)
        self.visualizer.show()