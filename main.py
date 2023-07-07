from environment_reader import EnvironmentReader
from window_interface import WindowInterface
import cv2

window_title = 'BlueStacks App Player'

if (__name__ == '__main__'):
    
    window_interface = WindowInterface()
    environment_reader = EnvironmentReader(window_interface)
    window_interface.update()
    window_interface.start_game()
    
    while True:
        window_interface.update()
        
        if (environment_reader.is_game_ended()):
            print("Round Ended")
            environment_reader.score = 0
            window_interface.restart_game(mode='caught')
        else:
            environment_reader.read_environment()
            environment_reader.visualize_window()
            # print(environment_reader.score)
        cv2.waitKey(8)