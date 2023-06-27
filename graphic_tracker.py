import cv2
import numpy as np
import pyautogui
import display
import keyboard
import time

# Set the region of interest (ROI) coordinates for capturing the game window
# Adjust these coordinates to match the position of the Tiny Wings game window on your screen
# The format is (left, top, width, height)
game_window_roi = (0, 0, 950, 550)
#display.make_window(game_window_roi)

# Set the parameters for hill detection and bird tracking
hill_threshold = 100  # Adjust this threshold based on the hills' visual characteristics
bird_template_path = 'bird.png'  # Path to the template image of the bird

# Load the bird template image
bird_template = cv2.imread(bird_template_path, 0)  # Load the image in grayscale
# Set the interval for generating points along the hills curve
point_interval = 10  # Adjust this value based on the desired density of the hill points

while True:
    # Capture the game window screenshot
    screenshot = pyautogui.screenshot(region=game_window_roi)
    screenshot = np.array(screenshot)
    
    # Preprocess the screenshot
    # Perform any necessary preprocessing steps like cropping, resizing, or color space conversions
    
    # Perform hill detection
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, hill_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process the detected contours and extract the hill points
    hill_points = []
    for contour in contours:
        # Process each contour and generate points along the curve
        perimeter = cv2.arcLength(contour, True)
        approx_curve = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        
        # Generate points along the curve at the specified interval
        for i in range(0, len(approx_curve), point_interval):
            point = approx_curve[i][0]
            hill_points.append(point)
    
    # Perform bird tracking
    bird_position = None
    # Use template matching or other techniques to locate the bird in the screenshot
    
    # Print the bird position and hill points
    print("Bird Position:", bird_position)
    print("Hill Points:", hill_points)
    display.visualize_points(game_window_roi, bird_position, hill_points)
    # Update the RL model with the extracted hill and bird information
    
    # Your RL model logic goes here
    
    # Exit the loop when the game is over or based on a specific condition
    
    # Break the loop for demonstration purposes
    cv2.waitKey(8)

# Clean up any resources used (e.g., release OpenCV windows, etc.)
cv2.destroyAllWindows()