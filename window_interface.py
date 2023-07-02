import pyautogui
import cv2
import numpy as np

# wa_top, wa_bottom, wa_left, wa_right = 0.08, 0.0, 0.17, 0.06
wa_top, wa_bottom, wa_left, wa_right = 0.08, 0.0, 0.0, 0.06

class WindowInterface:
    def __init__(self, window_title='BlueStacks App Player'):    
        self.window_title = window_title
        self.window = pyautogui.getWindowsWithTitle(window_title)[0]
        
        self.screenshot_id = 0
        self.steps = 0
        self.window_pressed = False
        
        self.screenshot = None

        self.window.activate()
        self.is_window_active = True

    def update(self):
        self.steps += 1
        active_window = pyautogui.getActiveWindow()
        self.is_window_active = active_window is not None and self.window_title in active_window.title
        
        if not self.is_window_active:
            return

        adjustment_top = int(self.window.height * wa_top)
        adjustment_bottom = int(self.window.height * wa_bottom)
        adjustment_left = int(self.window.width * wa_left)
        adjustment_right = int(self.window.width * wa_right)

        # Calculate the adjusted region
        self.window_roi = (
            self.window.left + adjustment_left,
            self.window.top + adjustment_top,
            self.window.width - adjustment_left - adjustment_right,
            self.window.height - adjustment_top - adjustment_bottom
        )

        self.window_center = (self.window_roi[0] + (0.5 * self.window_roi[2]), self.window_roi[1] + (0.5 * self.window_roi[3]))
    
        self.screenshot = pyautogui.screenshot(region=self.window_roi)

    def save_screenshot(self, save_root='screenshots/', num_steps = None):
        if num_steps != None and self.steps % num_steps != 0:
            return
        self.screenshot.save(save_root + str(self.screenshot_id) + ".png")
        self.screenshot_id += 1

    def update_window_press(self, is_window_pressed):

        if not self.is_window_active or self.window_pressed == is_window_pressed:
            return
        
        if is_window_pressed:
            pyautogui.mouseDown(self.window_center)
        else:
            pyautogui.mouseUp()

        self.window_pressed = is_window_pressed

    def restart_game(self, mode='default'):
        
        if not self.is_window_active:
            return

        if mode == 'caught':
            x,y = self.window_roi[0] + (0.68 * self.window_roi[2]), self.window_roi[1] + (0.87 * self.window_roi[3])
            pyautogui.click(x,y)
        
        if mode == 'default':
            x,y = self.window_roi[0] + (0.98 * self.window_roi[2]), self.window_roi[1] + (0.96 * self.window_roi[3])
            pyautogui.click(x,y)

            x,y = self.window_roi[0] + (0.19 * self.window_roi[2]), self.window_roi[1] + (0.49 * self.window_roi[3])
            pyautogui.click(x,y)

            x,y = self.window_roi[0] + (0.55 * self.window_roi[2]), self.window_roi[1] + (0.62 * self.window_roi[3])
            pyautogui.click(x,y)