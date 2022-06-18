#!/usr/bin/env python
# -*- coding: utf-8 -*-
import darknet
import darknet_images
import cv2
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
#from sensor.webcam import webcam
import webcam

#경로설정
PATH = os.path.abspath(os.path.dirname(__file__))
configPath  = os.path.join(PATH,"yolov4-macaron1-labeling12.cfg")
dataPath    = os.path.join(PATH,"customData/obj.data")
namePath    = os.path.join(PATH,"customData/obj.names")
weightPath  = os.path.join(PATH,"backup/track_3000.weights")


def main():
    import rospy
    import array
    import time
    import os

    network, class_names, class_colors = darknet.load_network(
        configPath,
        dataPath,
        weightPath
    )
    file_list = os.listdir('./corn_test')
    i = 0
    k = len(file_list)
    while i<k:
        # FOR TEST
        
        img = cv2.imread('./corn_test/'+file_list[i])
        
        image, detections= darknet_images.image_detection_cv2file(
            img, network, class_names, class_colors, .35
        )

        
        #cv2.imwrite('./test_img.jpg', test_img)
        #detections는 이미지에서 탐지된 것들의 list
        #그 안에는 탐지된 classnum, draw box의 각 꼭짓점의 값
        #message type에 맞게 설정
        #여러개를 인식했을 수도 있기 때문에 list로 해서 list를 message의 각 값으로 함
        #detections = [[label,prob,[box,,..]],[label....]]
        #cv2.imwrite('./corn_result/original.jpg', image)
        img_color = image
        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        lower_blue = (120-10, 30, 30)
        upper_blue = (120+10, 255,255)
        img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
        print(img_mask)
        cv2.imwrite('./result/'+str(i)+str(i)+'.jpg', img_mask)
        for label, prob, box in detections:
            left, top, right, bottom = darknet.bbox2points(box)
            blue_count = 0
            count = 0
            for l in range(top, bottom):
                for m in range(left,right):
                    count += 1
                    if img_mask[l][m] == 255:
                        blue_count += 1
            print('blue : ', blue_count, 'count : ',count)
            cv2.rectangle(image, (left, top), (right, bottom), (0,0,255), 1)
            if blue_count*2 > count:
                cv2.putText(image, "{}".format('b'),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,0,255), 2)

            elif blue_count*2 < count:
                cv2.putText(image, "{}".format('y'),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,0,255), 2)
            else:
                cv2.putText(image, "{}".format('error'),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,0,255), 2)


        cv2.imwrite('./result/'+str(i)+'.jpg', image)
        i+=1
    

if __name__ == "__main__":
    main()
