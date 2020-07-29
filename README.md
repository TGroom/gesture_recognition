
#Gesture Recognition: Waving
This package uses RGB camera video  for real-time gesture recognition, at the moment this package only detects a waving gesture.

##Installation
 Follow the tutorial [here](https://industrial-training-master.readthedocs.io/en/melodic/_source/session1/Installing-Existing-Packages.html#download-and-build-a-package-from-source) about downloading and building a package from github or follow the steps below:
 
1\. Clone gesture_recognition from github into your workspace:
`cd ~/catkin_ws/src`
`git clone https://github.com/---/---/git`

2\. Then in your workspace directory run:
`catkin build`

3\. Finally re-source your workspace using the following command:
`source ~/catkin_ws/devel/setup.bash`

##Usage
This package will subscribe to the topic`/xtion/rgb/image_raw/compressed` and set a `gesture` param in the ros parameter server with the result. 

To run the code, fisrt make sure that the ros sim is running and that the CompressedImage topic has data being published, then use **`roslaunch gesture_recognition waving.launch`**.

To change which topic the code retrieves the compressed image from, change the line in `node.py` accordingly:
```python
topicName = "<Topic Name e.g. /camera/image/compressed >"
```
Executing  `rosrun gesture_recognition waving.py` will use a webcam as input and print the output to the terminal to be used for testing and debugging.
##About

```python
gestures = {'wave': waving, 'hand_pos': (int(average_pos[0]-frame_shape[1]/2),int(average_pos[1]-frame_shape[0]/2))}
rospy.set_param('gestures', gestures)
```
`gesture/wave` is a bool assigned either True or False. This parameter is True when somone is detected waving.
`gesture/hand_pos` is a tuple containing two integers representing the x (horizontal) and y (vertical) averaged positions of the detected hand. The unit is in pixels and relative to the center of the frame [0,0]. A negative x component means that the hand is to the left of the frame/camera and vice versa.

##Requirements
This package requires the following python libraries:

 * OpenCV 3
 * Numpy
 
##References

Python CompressedImage Subscriber: http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
 

## License
[MIT](https://choosealicense.com/licenses/mit/)