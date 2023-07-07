from bird_tracker import BirdTracker
from game_end_tracker import GameEndTracker
from score_tracker import ScoreTracker
from visualizer import Visualizer
import cv2
import numpy as np

hill_threshold = 220

# Stored as relative indentations
# Left, Right, Top, Bottom
score_region = 0.871, 0.025, 0.01, 0.925
score_num_digits = 4

class EnvironmentReader:
    def __init__(self, window_interface):
        self.visualizer = Visualizer("Bird Visualization", hide_image=True)
        self.bird_tracker = BirdTracker()
        self.game_end_tracker = GameEndTracker()
        self.score_tracker = ScoreTracker()
        self.window_interface = window_interface
        self.step = 0
        self.screenshot_id = 1

        self.hill_points = None
        self.bird_point = None
        self.score = 0

    def read_environment(self):
        self.visualizer.set_image_array(self.window_interface.screenshot_array)
        self.update_hill_points()
        self.update_bird_point()
        self.update_score()

    def update_hill_points(self):

        screenshot_array = self.window_interface.screenshot_array

        point_threshold_y = 0.95 * self.window_interface.window_roi[3]

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

        self.hill_points = [point[1] for point in hill_points]
        self.visualizer.set_hills(self.hill_points)

    def update_bird_point(self):
        self.bird_coords = self.bird_tracker.get_coords(self.window_interface.screenshot)
        self.visualizer.set_bird(self.bird_coords)

    def update_score(self):
        
        screenshot = self.window_interface.screenshot

        width, height = screenshot.size
        sa_left, sa_right, sa_top, sa_bottom = score_region
        
        score_roi_left = int(sa_left * width)
        score_roi_top = int(sa_top * height)
        score_roi_right = int((1-sa_right) * width)
        score_roi_bottom = int((1-sa_bottom) * height)
        score_roi = (score_roi_left, score_roi_top, score_roi_right, score_roi_bottom)
        
        score_screenshot = screenshot.crop(score_roi)

        score = ''
        digit_width = int(score_screenshot.size[0]/score_num_digits)

        for i in range(score_num_digits):
            digit_region = (i*digit_width, 0, (i+1) * digit_width, score_screenshot.size[1])
            digit_screenshot = score_screenshot.crop(digit_region)
            score += str(self.score_tracker.get_score(digit_screenshot))
            self.screenshot_id += 1
        
        self.visualizer.set_score(score, (score_roi_left, score_roi_bottom))

        score = int(score)
        if score < self.score:
            #print("WARN: Score decreased")
            pass
        elif score - self.score > 50:
            #print("WARN: Score increase too high")
            pass
        else:
            self.score = score
            #print(score)

    def is_game_ended(self):
        return self.game_end_tracker.is_game_ended(self.window_interface.screenshot)

    def visualize_window(self):
        self.visualizer.show()