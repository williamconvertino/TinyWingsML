import pyautogui
import cv2
import numpy as np
from bird_tracker import BirdTracker
from visualizer import Visualizer

wa_top, wa_bottom, wa_left, wa_right = 0.08, 0.0, 0.17, 0.06

hill_threshold = 220

class WindowReader:
    def __init__(self, window_title):
        self.window = pyautogui.getWindowsWithTitle(window_title)[0]
        self.window.activate()
        self.window_title = window_title
        self.is_window_active = True
        self.bird_tracker = BirdTracker()
        self.visualizer = None
        self.screenshot_id = 1
        self.steps = 0
        self.window_pressed = False

    def scan_window(self):

        active_window = pyautogui.getActiveWindow()
        self.is_window_active = active_window is not None and self.window_title in active_window.title
        
        if not self.is_window_active:
            return

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

        self.window_center = (self.window_roi[0] + (0.5 * self.window_roi[2]), self.window_roi[1] + (0.5 * self.window_roi[3]))
        
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

    def save_images(self, save_path="screenshots/", num_steps = None):
        if num_steps != None and not self.steps % num_steps == 0:
            return
        self.screenshot.save(save_path + str(self.screenshot_id) + ".png")
        self.screenshot_id += 1

    def update_window_press(self, is_window_pressed):

        if self.window_pressed == is_window_pressed:
            return
        
        if is_window_pressed and self.is_window_active:
            pyautogui.mouseDown(self.window_center)
        else:
            pyautogui.mouseUp()

        self.window_pressed = is_window_pressed

    def restart_game(self, mode='default'):
        
        if not self.is_window_active:
            return
        
        pyautogui.mouseUp()

        if mode == 'caught':
            x,y = self.window_roi[0] + (0.68 * self.window_roi[2]), self.window_roi[1] + (0.87 * self.window_roi[3])
            pyautogui.click(x,y)
        
        if mode == 'default':
            x,y = self.window_roi[0] + (0.98 * self.window_roi[2]), self.window_roi[1] + (0.96 * self.window_roi[3])
            pyautogui.click(x,y)

            x,y = self.window_roi[0] + (0.19 * self.window_roi[2]), self.window_roi[1] + (0.49 * self.window_roi[3])
            pyautogui.click(x,y)

            x,y = self.window_roi[0] + (0.55 * self.window_roi[2]), self.window_roi[1] + (0.62 * self.window_roi[3])
            pyautogui.click(x,y)
            
    def vizualize_window(self, hide_image=False):
        if self.visualizer == None:
            self.visualizer = Visualizer("Bird Visualization")

        self.visualizer.set_image_array(self.screenshot_array, hide_image=hide_image)
        self.visualizer.set_bird(self.bird_coords)
        self.visualizer.set_hills(self.hill_points)
        self.visualizer.show()

def linear_interpolation(y1, y2, x):
        return y1 + (y2 - y1) * x