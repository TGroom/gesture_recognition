#!/usr/bin/python
from __future__ import print_function
import math
import numpy as np
import cv2
import rospy
import sys
from functions import getFrame, normalDistribution, detectWaving, getFrameInit
from sensor_msgs.msg import CompressedImage


class gesture:
    def __init__(self):
        # subscribed Topic
        global firstTry
        firstTry = True
        self.subscriber = rospy.Subscriber("/xtion/rgb/image_raw/compressed",
            CompressedImage, self.callback,  queue_size = 1)

    def callback(self, data):
        global firstTry
        np_arr = np.fromstring(data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if firstTry == True:
            getFrameInit(image_np)
            firstTry = False

        gest = detectWaving(getFrame(image_np))
        
        try:
            rospy.set_param('gesture', gest)  # Setting 'gestures' parameter in the parame server
            print(rospy.get_param('gesture'))
        except:
            print("Could not set 'gestures' param - Is roscore running?") 

def main(args):
    ic = gesture()
    rospy.init_node('gestures', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main(sys.argv)
