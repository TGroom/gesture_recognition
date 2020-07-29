#!/usr/bin/python
from __future__ import print_function
import numpy as np
import cv2
from functions import normalDistribution, getFrameInit, getFrame

positions = []
relative_pos = []
recent = []

def detectWaving(frameData):
    img, mask, mask_features, hand_pos, palms, fists, count = frameData
    frame_shape = img.shape[0:2]
   
    # Evaluating Motion:
    positions.append([hand_pos[0], hand_pos[1]])
    average_pos = np.average(positions, axis=0)
    relative_pos.append(positions[0][0] - average_pos[0])
    pos_range = max(positions[:][0]) - min(positions[:][0])
    if len(positions) > 5:
        positions.pop(0)
        relative_pos.pop(0)

    # List 'relative_pos' contains the relative displacement of the hand 
    # from an average position of the hand over 5 frames. The largest 
    # displacement to the left and right is mapped onto a normal distribution.
    max_factor = normalDistribution(max(max(relative_pos),0),20) 
    min_factor = normalDistribution(min(min(relative_pos),0),20)

    wave = 0
    if count < 5:   
        wave = max_factor*min_factor

    # Averaging Gestures:
    recent.append(wave) 
    average = np.average(np.asarray(recent))
    if len(recent) > 2:  # For faster detection, reduce the max length
        recent.pop(0)

    # This threashold is set between 0 and 1 to set the sensitivity of the detection
    threashold = 0.7
    waving = False
    if average > threashold:
        waving = True

    gestures = {'wave': waving, 'hand_pos': (int(average_pos[0]-frame_shape[1]/2),int(average_pos[1]-frame_shape[0]/2))}

    return gestures
    


if __name__ == '__main__':  # Plotting Image:
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()  # Get first frame
    getFrameInit(img)

    while(True):
        try:
            ret, img = cap.read()  # Get the current frame
            frameData = getFrame(img)
            img, mask, mask_features, hand_pos, palms, fists, count = frameData
            cv2.imshow('img', cv2.add(img, mask))
            print(detectWaving(frameData))
        except:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
