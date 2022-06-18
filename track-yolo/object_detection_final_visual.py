#!/usr/bin/env python
# -*- coding: utf-8 -*-
import darknet
import darknet_images
import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from macaron_3.msg import Traffic, obj_info
import webcam

#경로설정
PATH = os.path.abspath(os.path.dirname(__file__))
configPath  = os.path.join(PATH,"fin/yolo-obj.cfg")
dataPath    = os.path.join(PATH,"fin/obj.data")
namePath    = os.path.join(PATH,"fin/obj.names")
weightPath  = os.path.join(PATH,"fin/yolo-obj_15000.weights")

def main():
    import rospy
    import array
    import time
    import os
    cam = webcam(2)
    rospy.init_node("object_dection")
    network, class_names, class_colors = darknet.load_network(
        configPath,
        dataPath,
        weightPath
    )
    obj_pub = rospy.Publisher('traffic_obj', Traffic, queue_size=1)
    #Traffic이라는 메시지 타입의 객체를 obj_msg라는 이름으로 생성
    obj_msg = Traffic()

    while True:
        #현재 카메라 화면 읽어오기
        img = cam.capture(2)
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
            left, top, right, bottom = darknet.bbox2points(box)
            detected_obj = obj_info()
            detected_obj.ns = label
            detected_obj.xmin = left
            detected_obj.ymin = top
            detected_obj.xmax = right
            detected_obj.ymax = bottom
            obj_msg.obj.append(detected_obj)
            print("result :", label, prob, box)
            cv2.rectangle(image, (left, top), (right, bottom), (0,0,255), 1)
            cv2.putText(image, "{} [{:.2f}]".format(label, float(prob)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,0,255), 2)

        obj_pub.publish(obj_msg)
        cv2.imshow('test', img)
        
        if cv2.waitKey(1) != -1:
            break

    cv2.destroyAllWindows()
    cam.close()

if __name__ == "__main__":
    main()
