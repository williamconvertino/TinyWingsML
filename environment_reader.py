from bird_tracker import BirdTracker
from game_end_tracker import GameEndTracker
from score_tracker import ScoreTracker
from visualizer import Visualizer
import cv2
import numpy as np

hill_threshold = 220

#sa_top, sa_bottom, sa_left, sa_right = 0.00, 0.92, 0.79, 0.02
sa_top, sa_bottom, sa_left, sa_right = 0.00, 0.92, 0.885, 0.02

class EnvironmentReader:
    def __init__(self):
        self.bird_tracker = BirdTracker()
        self.game_end_tracker = GameEndTracker()
        self.score_tracker = ScoreTracker()
        self.step = 0
        self.screenshot_id = 1

        self.visualizer = None
        self.screenshot_array = None
        self.hill_points = None
        self.bird_point = None
        self.score = 0

    def update(self, screenshot, window_roi):
        self.screenshot_array = np.array(screenshot)
        self.window_roi = window_roi
        self.screenshot = screenshot
        self.step += 1

    def read_environment(self):
        self.update_hill_points()
        self.update_bird_point()
        self.update_score()

    def update_hill_points(self):

        point_threshold_y = 0.95 * self.window_roi[3]

        height, width, _ = self.screenshot_array.shape

        hill_points = [(i, height) for i in range(width)]

        grayscale_image = cv2.cvtColor(self.screenshot_array, cv2.COLOR_BGR2GRAY)
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

    def update_bird_point(self):
        self.bird_coords = self.bird_tracker.get_coords(self.screenshot)

    def update_score(self):
        
        width, height = self.screenshot.size

        score_roi_left = int(sa_left * width)
        score_roi_top = int(sa_top * height)
        score_roi_right = int((1-sa_right) * width)
        score_roi_bottom = int((1-sa_bottom) * height)

        num_digits = 4
        digit_width = (score_roi_right-score_roi_left)/num_digits
        
        score = ''

        for i in range(num_digits):
            score_roi = (score_roi_left + (i * digit_width), score_roi_top, score_roi_left + ((i+1) * digit_width), score_roi_bottom)
            score += str(self.score_tracker.get_score(self.screenshot.crop(score_roi)))
            self.screenshot_id += 1
        
        score = int(score)
        if score >= self.score:
            self.score = score
        else:
            print("Score has been miscalculated.")

    def is_game_ended(self):
        return self.game_end_tracker.is_game_ended(self.screenshot)

    def visualize_window(self, hide_image=False):
        if self.visualizer == None:
            self.visualizer = Visualizer("Bird Visualization")

        self.visualizer.set_image_array(self.screenshot_array, hide_image=hide_image)
        self.visualizer.set_bird(self.bird_coords)
        self.visualizer.set_hills(self.hill_points)
        self.visualizer.show()