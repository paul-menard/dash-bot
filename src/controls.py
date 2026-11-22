### the python scripts that handles the controls (especially the jump mechanism) based on the order from logic.py

from concurrent.futures import ThreadPoolExecutor
import pydirectinput
import time

pydirectinput.PAUSE = 0 # disable the default delay inputs, this caused massive
## fps drops

## the controls that can be asked from the program are basically to jump whatevers happens
## we could decide to add more controls in the future like, letting the program auto launch the game 
## or some other shit like this
## but for now on only jumps ahah

executor = ThreadPoolExecutor(max_workers=2)

def click():
    executor.submit(pydirectinput.click)

def _long_click_task(alpha):
    pydirectinput.keyDown('space')
    time.sleep(alpha * 0.01)
    pydirectinput.keyUp('space')

def long_click(alpha=5):
    executor.submit(_long_click_task, alpha)

def stationary_click_task():
    time.sleep(0.03)
    pydirectinput.keyDown('space')
    time.sleep(0.022)
    pydirectinput.keyUp('space')

def stationary_click():
    executor.submit(stationary_click_task)

def open_key():
    pydirectinput.keyDown('space')

def close_key():
    pydirectinput.keyUp('space')