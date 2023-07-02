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

    def read(self, screenshot, window_roi):
        self.screenshot_array = np.array(screenshot)
        self.window_roi = window_roi

        self.update_hill_points(self.screenshot_array)
        self.update_bird_point(screenshot)

    def update_hill_points(self, screenshot_array):

        
        point_threshold_y = 0.95 * self.window_roi[3]

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

        self.hill_points = []

        left_point = None
        left_point_index = 0
        right_point = None
        right_point_index = 0

        #Find left point
        for i in range(len(hill_points)):
            point = hill_points[i]
            if point[1] < point_threshold_y:
                left_point = point
                left_point_index = i
                break
        
        if left_point == None:
            return

        #Fill in previous points
        for i in range(left_point_index):
            hill_points[i] = (hill_points[i][0], left_point[1])

        #Find right point
        for i in range(len(hill_points)-1, 0, -1):
            point = hill_points[i]
            if point[1] < point_threshold_y:
                right_point = point
                right_point_index = i
                break
        
        if right_point == None:
            return

        #Fill in previous points
        for i in range(left_point_index, 0, -1):
            hill_points[i] = (hill_points[i][0], right_point[1])

        #Start loop
        for i in range(left_point_index, len(hill_points)):
            point = hill_points[i]
            if point[1] < point_threshold_y:
                left_point = point
                continue
            right_point = hill_points[-1]
            right_point_index = i + 1
            while (right_point_index < len(hill_points)):
                right_point = hill_points[right_point_index]
                if right_point[1] < point_threshold_y:
                    break
                right_point_index += 1
            
            for j in range(i, right_point_index):
                hill_points[j] = (hill_points[j][0], int((left_point[1] + right_point[1]) / 2))

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