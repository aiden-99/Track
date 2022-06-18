#!/usr/bin/env python
# -*- coding: utf-8 -*-
import darknet
import darknet_images
import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pyzed.sl as sl
import numpy as np
from macaron_3.msg import Traffic, obj_info

#경로설정
PATH = os.path.abspath(os.path.dirname(__file__))
configPath  = os.path.join(PATH,"yolov4-macaron1-labeling12.cfg")
dataPath    = os.path.join(PATH,"customData/obj.data")
namePath    = os.path.join(PATH,"customData/obj.names")
weightPath  = os.path.join(PATH,"backup/yolov4-macaron1-labeling12_4000.weights")

def main():
    import rospy
    import array
    import time
    import os
    
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode

    image_size = zed.get_camera_information().camera_resolution
    image_size.width = 640
    image_size.height = 480

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
        image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)

        err = zed.grab(runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS :
        # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            img = image_zed.get_data()

        image, detections = darknet_images.image_detection_cv2file(
            img, network, class_names, class_colors, .35
        )
        
        #detections는 이미지에서 탐지된 것들의 list
        #그 안에는 탐지된 classnum, draw box의 각 꼭짓점의 값
        #message type에 맞게 설정
        #여러개를 인식했을 수도 있기 때문에 list로 해서 list를 message의 각 값으로 함
        #detections = [[label,prob,[box,,..]],[label....]]
        for label, prob, box in detections:
            detected_obj = obj_info()
            detected_obj.ns = label
            left, top, right, bottom = darknet.bbox2points(box)
            detected_obj.xmin = left
            detected_obj.ymin = top
            detected_obj.xmax = right
            detected_obj.ymax = bottom
            obj_msg.obj.append(detected_obj)
            cv2.rectangle(image, (left, top), (right, bottom), (0,0,255), 1)
            cv2.putText(image, "{} [{:.2f}]".format(label, float(prob)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,0,255), 2)

        obj_pub.publish(obj_msg)
        cv2.imshow('test', image)
        
        if cv2.waitKey(1) != -1:
            break
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
