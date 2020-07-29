#!/usr/bin/python
from __future__ import print_function
import math
import sys
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from functions import getFrame, normalDistribution, getFrameInit
from waving import detectWaving


class gesture:
    def __init__(self):
        # subscribed Topic
        global firstTry
        firstTry = True

        topicName = "/xtion/rgb/image_raw/compressed"

        self.subscriber = rospy.Subscriber(topicName,
            CompressedImage, self.callback,  queue_size = 1)

        print("Subscribed to:" + topicName)

    def callback(self, data):
        global firstTry
        np_arr = np.fromstring(data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if firstTry == True:
            getFrameInit(image_np)  # Initialise getFrame() variables
            firstTry = False

        frameData = getFrame(image_np)  # Getting the frame
        gest = detectWaving(frameData)  # Running the waving code

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
