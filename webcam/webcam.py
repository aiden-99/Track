#!/usr/bin/env python

import cv2

class webcam:
    #camera number = 5?
    def __init__(self, index = 4):
        cap = cv2.VideoCapture(index)
        
        cap.set(3, 640)#width
        cap.set(4, 480)#height
        self.cap = cap

    def capture(self):
        _, img = self.cap.read()
        return img #image read


def main():
    import rospy
    rospy.init_node("webcam_test")
    cam = webcam()
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        img = cam.capture()
        cv2.imshow("result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rate.sleep()
## start code
if __name__ == '__main__':
    main()
