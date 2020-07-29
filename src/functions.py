#!/usr/bin/python
from __future__ import print_function
import math
import os
import numpy as np
import cv2

# Defining the directory of the Haar Cascades:
file_location = os.path.dirname(os.path.realpath(__file__))
palm_cascade = cv2.CascadeClassifier(file_location+'/haarcascade_palm.xml')	
fist_cascade = cv2.CascadeClassifier(file_location+'/haarcascade_fist.xml')

# Defining parameters for the optical flow:
feature_params = dict(maxCorners = 200, qualityLevel = 0.3, minDistance = 5, blockSize = 5 )
lk_params = dict( winSize  = (15,15), maxLevel = 2,
				  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
hand_pos = [320,240]
count = 0


def getFrameInit(img):
    global p0    
    global old_gray
    global mask_features

    old_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    mask_features = np.zeros(old_gray.shape, dtype=np.uint8)

def getFrame(img):
    """ This function returns a frame and applies haarcascades and 
    optical flow for further analysis
    """
    global p0    
    global old_gray
    global hand_pos
    global count
    global mask_features

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame_shape = gray.shape
    mask = np.zeros_like(img)

    # Haarcascade Detection:
    try:
        palms = palm_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 15, minSize = (20,20))
        fists = fist_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 13, minSize = (40,40)) 
    except cv2.error:
        print("Could not find 'haarcascade_fist.xml' and/or 'haarcascade_palm.xml' file in: "+file_location+'/')

    if np.any(fists):
        x,y,w,h = fists[0]
        hand_shape = w*1.8

    if np.any(palms):
        multi_hand_dist = []
        for x,y,w,h in palms:  # For every hand detected:
            multi_hand_dist.append(np.hypot(hand_pos[0]-(x+(w/2)), hand_pos[1]-(y+(h/2))))
            cv2.rectangle(mask, (x, y), (x+w, y+h), (0,40,250), 1)
            cv2.circle(mask, (int(x+(w/2)), int(y+(h/2))), int(w/1.7), color=(0,0,0), thickness=-1)
        x,y,w,h = palms[np.argmin(multi_hand_dist)]	 # Use the detected hand which is closest to the previously detected hand
        hand_shape = w
        count = 0

    if np.any(palms) or np.any(fists):
        hand_pos = np.array([int(x+(w/2)), int(y+(h/2))])  # Defines the new hand position as the center of the detected hand
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0,255,0), 1)
        cv2.circle(mask,tuple(hand_pos), int(w/1.8), color=(0,0,0), thickness=-1)
        mask_features = np.zeros(frame_shape, dtype=np.uint8)  # Masking out the hand for goodFeaturesToTrack
        cv2.circle(mask_features,tuple((hand_pos[0], hand_pos[1]-10)), int(hand_shape/1.8), color=1, thickness=-1)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask_features, **feature_params)
    
    # Optical Flow:    
    elif p0 is not None:  # If no hand/fist was detected, use optical flow to give an estimate for the location of the hand
        count += 1
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
        if np.sum(st) > 0:  # If it found at least 1 good new point
            good_new = p1[st==1]
            good_old = p0[st==1]
            p0 = good_new.reshape(-1, 1, 2)
            flows = []
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                cv2.circle(mask, (a, b), 2, [0,255,0], -1)
                flows.append([a-c,b-d])
            flow = np.average(flows, 0)  # Average all movements of every tracked point
            hand_pos = np.add(hand_pos, flow)  # Update the hand position
            hand_pos = np.array([int(hand_pos[0]), int(hand_pos[1])])
            
    cv2.circle(mask, tuple(hand_pos), 4, (0,255,0), -1)
    old_gray = gray.copy()
    
    return img, mask, mask_features, hand_pos, palms, fists, count

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def cart2pol(coordinate, center):
    x,y = (coordinate[0]-center[0],coordinate[1]-center[1])
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (r, theta)


def pol2cart(r, theta, center):
    x = r*np.cos(theta) + center[0]
    y = r*np.sin(theta) + center[1]
    return (int(x), int(y))


def normalDistribution(val, mag):
    return 1-math.exp((-val**2) / mag)


def floodfill_mask(input_img, frame_shape, point, lo, hi):
    fill_params = dict(newVal=(0,0,0,0), flags = 8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | ( 255 << 8 ))	
    point = [min(max(int(point[0]), 0), frame_shape[1]-1),
             min(max(int(point[1]), 0), frame_shape[0]-1)]  # To avoid points which are outside of the frame
    h, w = frame_shape
    flood_mask = np.zeros((h+2, w+2), dtype=np.uint8)
    cv2.floodFill(input_img, flood_mask, seedPoint=tuple(point), loDiff=(lo,lo,lo,lo), upDiff=(hi,hi,hi,hi), **fill_params)
    return flood_mask[1:-1, 1:-1]


