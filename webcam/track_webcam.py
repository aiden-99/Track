#!/usr/bin/env python
# -*- coding: utf-8 -*-
import darknet
import darknet_images
import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import webcam
import numpy as np
from macaron_3.msg import Traffic, obj_info

#경로설정
PATH = os.path.abspath(os.path.dirname(__file__))
configPath  = os.path.join(PATH,"yolov4-macaron1-labeling12.cfg")
dataPath    = os.path.join(PATH,"customData/obj.data")
namePath    = os.path.join(PATH,"customData/obj.names")
weightPath  = os.path.join(PATH,"backup/yolov4-macaron1-labeling12_3000.weights")

def main():
    import rospy
    import array
    import time
    import os
    cam = webcam.webcam(1)

    rospy.init_node("object_dection")
    network, class_names, class_colors = darknet.load_network(
        configPath,
        dataPath,
        weightPath
    )
    obj_pub = rospy.Publisher('traffic_obj', Traffic, queue_size=1)
    #Traffic이라는 메시지 타입의 객체를 obj_msg라는 이름으로 생성
    obj_msg = Traffic()

    # TEST END
    '''
    cur_time = time.time()
    os.mkdir('/home/macaron/test_run/' + str(cur_time))
    i = 0
    #testimg = ['','2','75','160','324','371','391','404','429','545','547']
    testimg = range(1,19)
    print(testimg)
    '''

    while True:
        # FOR TEST
        
        """
        i+=1
        print(base_path + imagePath[i])
        img = cv2.imread(base_path + imagePath[i])
        """
        # TEST END
        #현재 카메라 화면 읽어오기
        img = cam.capture()
        #try:
        #   filename = str(testimg[i])
        #   print(filename)
        #except :
        #    break
        #img = cv2.imread("/home/macaron/catkin_ws/src/macaron_2/src/darknet/t/" + filename + ".jpg")
        #이거 새로 만든 함수 인듯--image_detection_cv2file
        image, detections = darknet_images.image_detection_cv2file(
            img, network, class_names, class_colors, .35
        )
        #원래 이미지에 탐지된거 box그려서 이미지 반환
        darknet.draw_boxes(detections, image, class_colors)
        
        classes = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        #detections는 이미지에서 탐지된 것들의 list
        #그 안에는 탐지된 classnum, draw box의 각 꼭짓점의 값
        #message type에 맞게 설정
        #여러개를 인식했을 수도 있기 때문에 list로 해서 list를 message의 각 값으로 함
        #detections = [[label,prob,[box,,..]],[label....]]
        for label, prob, box in detections:
            detected_obj = obj_info()
            detected_obj.ns = label
            #classes += label
            detected_obj.xmin = (int(box[0]))
            detected_obj.ymin = (int(box[1]))
            detected_obj.xmax = (int(box[2]))
            detected_obj.ymax = (int(box[3]))
            obj_msg.obj.append(detected_obj)
            print("result :", label, prob, box)
            xmin =int(box[0])+int(int(box[2]))
            ymin = int(box[1]) - int(int(box[3]))
            xmax = int(box[0])+int(int(box[2]))*3
            ymax = int(box[1])+int(int(box[3])/2)
            img = cv2.line(img, (xmin, ymin), (xmin, ymax), (0,0,255))
            img = cv2.line(img, (xmin, ymax), (xmax, ymax), (0,0,255))
            img = cv2.line(img, (xmax, ymax), (xmax, ymin), (0,0,255))
            img = cv2.line(img, (xmax, ymin), (xmin, ymin), (0,0,255))
            img = cv2.putText(img, label, (xmin, ymin-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 2)
            #img = cv2.line(img, (int(box[0])+i, int(box[1])+j), (int(box[0])+i, int(box[3])+j), (0,0,255))
            #img = cv2.line(img, (int(box[0])+i, int(box[3])+j), (int(box[2])+i, int(box[3])+j), (0,0,255))
            #img = cv2.line(img, (int(box[2])+i, int(box[3])+j), (int(box[2])+i, int(box[1])+j), (0,0,255))
            #img = cv2.line(img, (int(box[2])+i, int(box[1])+j), (int(box[0])+i, int(box[1])+j), (0,0,255))
            #img = cv2.putText(img, label, (int(box[0]), int(box[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 2)
        #image = ObjectDetection.draw_boxes(img, detection)
        #print("i, length", i, len(detections))
        #obj_msg.Class = classes
        #obj_msg.xmin = xmin
        #obj_msg.ymin = ymin
        #obj_msg.xmax = xmax
        #obj_msg.ymax = ymax

        obj_pub.publish(obj_msg)
        cv2.imshow('test', img)
        
        if cv2.waitKey(1) != -1:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
