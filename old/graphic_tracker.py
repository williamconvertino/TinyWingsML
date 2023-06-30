import cv2
import numpy as np
import pyautogui
import display
import keyboard
import time
from track_bird_model import ObjectPositionPredictor
import pytesseract

# Set the region of interest (ROI) coordinates for capturing the game window
# Adjust these coordinates to match the position of the Tiny Wings game window on your screen
# The format is (left, top, width, height)

#left, top, width, height = 0, 25, 825, 500
left, top, width, height = 0, 25, 825, 500
game_window_roi = (left, top, width, height)
#display.make_window(game_window_roi)

height_threshold = height * 0.2
height_min_threshold = height * 0.9
hill_threshold = 220  # Adjust this threshold based on the hills' visual characteristics

# Set the parameters for hill detection and bird tracking
bird_template_path = 'bird.png'  # Path to the template image of the bird
bird_template = cv2.imread(bird_template_path, 0)  # Load the image in grayscale

bird_boundaries_right = width * 0.3
bird_boundaries_left = 0


bird_predictor = ObjectPositionPredictor('bird_detector.pth')

def read_score(screenshot_array):

    x, y, w, h = (600,20,220,30)

    roi_image = screenshot_array[y:y+h, x:x+w]
    
    # Convert the screenshot to grayscale
    grayscale = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    # Read the number using Tesseract OCR
    score = pytesseract.image_to_string(grayscale, config='digits')

    print(score)
    return score

def linear_interpolation(y1, y2, x):
    return y1 + (y2 - y1) * x

def capture():

    id = 1

    while True:
        # Capture the game window screenshot
        screenshot = pyautogui.screenshot(region=game_window_roi)
        
        bird_position = bird_predictor.predict_single_image(screenshot)
        bird_position = (int(bird_position[0]),int(bird_position[1]))
        

        
        # screenshot.save('captures/' + str(id) + ".png")
        # id += 1

        screenshot = np.array(screenshot)

        read_score(screenshot)

        # Preprocess the screenshot
        # Perform any necessary preprocessing steps like cropping, resizing, or color space conversions
        
        # Perform hill detection
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, hill_threshold, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        hill_coordinates = [(i, height) for i in range(width)]
        
        for contour in contours:

            # # Approximate the contour to reduce the number of points
            # epsilon = 0.01 * cv2.arcLength(contour, True)
            # approx_contour = cv2.approxPolyDP(contour, epsilon, True)

            # Add the contour points to the hill coordinates
            for point in contour:
                x, y = point[0]
                if y > height_threshold and hill_coordinates[x][1] > y:
                    hill_coordinates[x] = (x, y)

        last_good_point = None
        for i in range(width):
            if hill_coordinates[i][1] > height_min_threshold:
                if last_good_point is not None:
                    hill_coordinates[i] = (i, last_good_point[1])
            else:
                last_good_point = hill_coordinates[i]

        #display.visualize_points(game_window_roi, bird_position, hill_coordinates)
        display.visualize_points(screenshot, bird_position, hill_coordinates)
        
        cv2.waitKey(8)

    # Clean up any resources used (e.g., release OpenCV windows, etc.)
    cv2.destroyAllWindows()

capture()