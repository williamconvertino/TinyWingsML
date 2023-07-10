from environment_reader import EnvironmentReader
from window_interface import WindowInterface
import cv2
from brain import Brain

window_title = 'BlueStacks App Player'

if (__name__ == '__main__'):
    
    window_interface = WindowInterface()
    environment_reader = EnvironmentReader(window_interface)
    window_interface.update()
    environment_reader.read_environment()               
    brain = Brain(len(environment_reader.hill_points)) #TODO: I'd love to remove the update/read lines, and have a constant window width

    while True:
        window_interface.update()
        environment_reader.read_environment()

        match environment_reader.game_state:
            case 'lose_waiting':
                brain.end_episode()
            case 'lose':
                window_interface.restart_game()
            case 'ready':
                window_interface.start_game()
                brain.start_episode()
            case 'pause':
                brain.end_episode()
                window_interface.restart_game('pause')
            case 'playing':
                # environment_reader.visualize_window()
                window_interface.update_window_press(brain.action)
                brain.update_reward(environment_reader.score)
                brain.add_observation(environment_reader.hill_points, environment_reader.bird_coords)
        cv2.waitKey(8)