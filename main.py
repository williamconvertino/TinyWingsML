from window_reader import WindowReader
import cv2
import pyautogui

window_title = 'BlueStacks App Player'

if (__name__ == '__main__'):
    
    window_reader = WindowReader(window_title)
    
    step = 0
    window_reader.scan_window()
    window_reader.save_images()
    window_reader.restart_game()
    do_window_press = True
    window_reader.scan_window()
    window_reader.update_window_press(do_window_press)
        
    # while True:

    #     print(f'Step: {step}')
    #     step += 1
    #     window_reader.scan_window()
    #     window_reader.save_images()
    #     window_reader.update_window_press(True)
    #     #window_reader.vizualize_window(hide_image=True)
    #     window_reader.update_window_press(do_window_press)
    #     do_window_press = not do_window_press
    #     cv2.waitKey(300)