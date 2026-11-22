### the code that launches everything !!

import cv2 as cv
import numpy as np
import time
from ultralytics import YOLO
import sys
import torch
sys.path.insert(1, './src')

from capture import WindowCapture
from logic import *

"""
import torch
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#exit()
"""

wincap = WindowCapture('Geometry Dash')
w = wincap.w
h = wincap.h
model = YOLO('runs/detect/train/weights/best.pt')
character = Cube(w, h) ## we always start the games as a cube :)
#wincap.list_window_names()

game_rect = wincap.get_window_rect()  # (left, top, right, bottom)
heatmap_window_name = "Rocket Cost Map Heatmap"
heatmap_win_created = False
BAR_W, LEGEND_W = 150, 70

valid_character_ids = torch.tensor([2, 5], device='cuda:0')


loop_time = time.time()
while True:

    screenshot = wincap.get_screenshot()
    #cv.imshow("Computer Vision", screenshot) 

    #print("FPS: {}".format(1/(time()-loop_time)))
    #loop_time = time()

    results = model.predict(screenshot, conf=0.55, verbose=False)
    cls_list = results[0].boxes.cls

    ## check the type of character, if not the right make the switch to the right one
    """
    is_type = character.check_type(results)
    if not is_type : 
        for cls in cls_list : 
            if cls in character_cls() :
                character = convert_to_cls(cls)
                print(character)
    """

    filter1 = cls_list == character_cls
    charac_cls = cls_list[filter1]
    if not charac_cls.any():
        matches = torch.isin(cls_list, valid_character_ids)
        if matches.any():
        # Get the first matching ID
        # .nonzero() finds indices where match is True
            new_cls_id = cls_list[matches.nonzero()[0]].item()
            character = convert_to_cls(new_cls_id, character.screen_w, character.screen_h)


    #print(character)
    """
    seems to be working so far, it seems like the code is able to change classes, there seems to be a problem 
    with the detection of the cubes though.....
    Furthermore, this is surely not the fastest method out there but we will come back to this later
    """

    ## now that we have checked the character, we need to let it behave as it should
    rectangles = results[0].boxes.xyxy
    cls = results[0].boxes.cls
    filter = cls == character.cls_id

    if len(rectangles[filter]) > 0:
        character_rectangle = rectangles[filter][0]
        character.rectangle = character_rectangle
    ## filter works well, maybe not optimized haha
        character.x, character.y = character.position(character_rectangle)

        if character.cls_id == 5:
            character.barycenter = character.get_barycenter(character_rectangle)
        #print(character.x, character.y)

    ## I guess it is ok to skip a few if the cube is not detected for a small amount of time
    character.behave(rectangles, cls)  ## depends on the rectangles and where they are.      
    ## character.behave should return a click or not depending on the situation
    ## the current behave method is too slow in dense zones :
    ## we need to filter out the data before 


    # plot() draws the boxes. 
    # labels=False removes the class name
    # conf=False removes the confidence score
    '''
    ## switching to a faster alternative here
    annotated_frame = results[0].plot(labels=True, conf=True)
    print(results[0].boxes)
    print(results[0].boxes.data)
    '''
    annotated_frame = results[0].plot(labels=True, conf=True)

    #height, width, channels = annotated_frame.shape
    #print(f"Taille de l'image : {width}x{height} pixels")
    #annotated_frame = cv.rectangle(screenshot, results[0].boxes)
    ### we will use a faster method later when all of this works
    cv.imshow("Computer Vision", annotated_frame) 
    #print("FPS: {}".format(1/(time.time()-loop_time)))
    loop_time = time.time()
    if character.cls_id == 5:
        heatmap = character.get_heatmap_frame(BAR_W, LEGEND_W)
        if heatmap is not None:
            if not heatmap_win_created:
                cv.namedWindow(heatmap_window_name, cv.WINDOW_NORMAL)
                cv.resizeWindow(heatmap_window_name, BAR_W + LEGEND_W, h)
                # place it just to the right of the game window
                cv.moveWindow(heatmap_window_name, game_rect[2] + 10, game_rect[1])
                heatmap_win_created = True
            cv.imshow(heatmap_window_name, heatmap)
    


    if cv.waitKey(1) == ord("q"): ## press q in the output window to exit, waits 1ms to process key presses
        cv.destroyAllWindows() ## destroys all of the windows
        break ## ends the loop


print("Done.")