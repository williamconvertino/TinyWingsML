from environment_reader import EnvironmentReader
from window_interface import WindowInterface
import cv2
from brain import Brain

window_title = 'BlueStacks App Player'

if (__name__ == '__main__'):
    
    window_interface = WindowInterface()
    environment_reader = EnvironmentReader(window_interface)
    window_interface.update()
    window_interface.start_game()
    environment_reader.read_environment()
    brain = Brain(len(environment_reader.hill_points))
    brain.start_episode()

    while True:
        window_interface.update()
        environment_reader.read_environment()
        
        if (environment_reader.is_game_ended()):
            print("Round Ended")
            environment_reader.score = 0
            window_interface.restart_game(mode='caught')
            brain.end_episode()
            brain.start_episode()
        else:
            environment_reader.visualize_window()
            brain.add_observation(environment_reader.hill_points, environment_reader.bird_coords)
            brain.update_reward(environment_reader.score)
            #window_interface.update_window_press(bool(brain.action))
        cv2.waitKey(8)