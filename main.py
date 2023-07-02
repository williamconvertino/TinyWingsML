from environment_reader import EnvironmentReader
from window_interface import WindowInterface
import cv2

window_title = 'BlueStacks App Player'

if (__name__ == '__main__'):
    
    window_interface = WindowInterface()
    environment_reader = EnvironmentReader()
    
    while True:
        window_interface.update()
        environment_reader.read(window_interface.screenshot, window_interface.window_roi)
        environment_reader.visualize_window()
        if (environment_reader.is_game_ended()):
            window_interface.restart_game(mode='caught')
        cv2.waitKey(100)