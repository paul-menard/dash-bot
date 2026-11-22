### the python script that handles the acquisition of information from the screen

import numpy as np
import win32gui
import ctypes
import mss

# following line helps if you put the videos in different folders
#os.chdir(os.path.dirname(os.path.abspath(__file__)))


#Tell Windows to provide physical pixel coordinates
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # 2 = PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    # Fallback for older Windows versions
    ctypes.windll.user32.SetProcessDPIAware()

class WindowCapture():

    ## properties
    w = 0
    h = 0
    hwnd = None
    OFFSET_X = 8
    OFFSET_Y = 31

    def __init__(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))
        rect = win32gui.GetClientRect(self.hwnd)
        self.w = rect[2] - rect[0] - (self.OFFSET_X * 2)
        self.h = rect[3] - rect[1] - self.OFFSET_Y - self.OFFSET_X
    
    
    def get_screenshot(self):
        with mss.mss() as sct:
            # Define the capture monitor area (using primary monitor 0)
            # We define the specific X,Y and W,H we calculated in __init__
            monitor_area = {
                "top": win32gui.GetWindowRect(self.hwnd)[1] + self.OFFSET_Y,
                "left": win32gui.GetWindowRect(self.hwnd)[0] + self.OFFSET_X,
                "width": self.w,
                "height": self.h
            }

            # Grab the raw pixels from the screen at this location
            sct_img = sct.grab(monitor_area)

            # Convert the MSS image directly to a NumPy array
            img = np.array(sct_img)

            # MSS returns BGRA (Blue, Green, Red, Alpha), but OpenCV expects BGR
            # So we drop the Alpha channel.
            img = img[:, :, :3]

        return img

    ## tested, works well
    def list_window_names(self):
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)
    
    def get_window_rect(self):
        """Returns (left, top, right, bottom) of the game window in screen coords."""
        return win32gui.GetWindowRect(self.hwnd)

