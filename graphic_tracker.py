import cv2
import numpy as np
import pyautogui
import display
import keyboard
import time
from scipy.interpolate import interp1d

# Set the region of interest (ROI) coordinates for capturing the game window
# Adjust these coordinates to match the position of the Tiny Wings game window on your screen
# The format is (left, top, width, height)

left, top, width, height = 0, 50, 900, 600
game_window_roi = (left, top, width, height)
#display.make_window(game_window_roi)

height_threshold = height * 0.2
hill_threshold = 220  # Adjust this threshold based on the hills' visual characteristics

# Set the parameters for hill detection and bird tracking
bird_template_path = 'bird.png'  # Path to the template image of the bird
bird_template = cv2.imread(bird_template_path, 0)  # Load the image in grayscale


while True:
    # Capture the game window screenshot
    screenshot = pyautogui.screenshot(region=game_window_roi)
    screenshot = np.array(screenshot)
    
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



    

    bird_position = None
    
    display.visualize_points(game_window_roi, bird_position, hill_coordinates)
    
    cv2.waitKey(800)

# Clean up any resources used (e.g., release OpenCV windows, etc.)
cv2.destroyAllWindows()