from environment_reader import EnvironmentReader
from window_interface import WindowInterface
import cv2
import pyautogui

window_title = 'BlueStacks App Player'

if (__name__ == '__main__'):
    
    window_interface = WindowInterface()
    environment_reader = EnvironmentReader()
    
    while True:
        window_interface.update()
        environment_reader.read(window_interface.screenshot)
        environment_reader.visualize_window()
        cv2.waitKey(8)