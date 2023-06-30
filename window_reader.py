import pyautogui
import cv2
import numpy as np
from bird_tracker import BirdTracker
from visualizer import Visualizer

window_title = 'BlueStacks App Player'
wa_top, wa_bottom, wa_left, wa_right = 0.08, 0.0, 0.17, 0.06

hill_threshold = 220

class WindowReader:
    def __init__(self, window_title=window_title):
        self.window = pyautogui.getWindowsWithTitle(window_title)[0]
        self.bird_tracker = BirdTracker()
        self.visualizer = None
        self.screenshot_id = 1
        self.steps = 0

    def scan_window(self):

        self.steps += 1

        adjustment_top = int(self.window.height * wa_top)
        adjustment_bottom = int(self.window.height * wa_bottom)
        adjustment_left = int(self.window.width * wa_left)
        adjustment_right = int(self.window.width * wa_right)

        # Calculate the adjusted region
        self.window_roi = (
            self.window.left + adjustment_left,
            self.window.top + adjustment_top,
            self.window.width - adjustment_left - adjustment_right,
            self.window.height - adjustment_top - adjustment_bottom
        )
        
        self.screenshot = pyautogui.screenshot(region=self.window_roi)
        self.screenshot_array = np.array(self.screenshot)

        self.find_bird_coords()
        self.find_hill_points()

    def find_hill_points(self):
        
        # max_point_value = self.window.height * 0.9
        # min_point_value = self.window.height * 0.2

        hill_points = [(i, self.window.height) for i in range(self.window.width)]
        
        grayscale_image = cv2.cvtColor(self.screenshot_array, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(grayscale_image, hill_threshold, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            for point in contour:
                x, y = point[0]
                if y < hill_points[x][1]: 
                    hill_points[x] = (x, y)

        self.hill_points = hill_points

    def find_bird_coords(self):
        self.bird_coords = self.bird_tracker.get_coords(self.screenshot)

    def save_images(self, num_steps, save_path="screenshots/"):
        if not self.steps % num_steps == 0:
            return
        self.screenshot.save(save_path + str(self.screenshot_id) + ".png")
        self.screenshot_id += 1

    def vizualize_window(self, hide_image=False):
        if self.visualizer == None:
            self.visualizer = Visualizer("Bird Visualization")

        self.visualizer.set_image_array(self.screenshot_array, hide_image=hide_image)
        self.visualizer.set_bird(self.bird_coords)
        self.visualizer.set_hills(self.hill_points)
        self.visualizer.show()

def linear_interpolation(y1, y2, x):
        return y1 + (y2 - y1) * x

if (__name__ == '__main__'):
    window_reader = WindowReader()

    while True:
        window_reader.scan_window()
        window_reader.save_images(5)
        window_reader.vizualize_window(hide_image=True)
        cv2.waitKey(8)