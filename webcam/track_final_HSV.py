#!/usr/bin/env python
# -*- coding: utf-8 -*-
import darknet
import darknet_images
import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pyzed.sl as sl
import math
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32

#경로설정
PATH = os.path.abspath(os.path.dirname(__file__))
configPath  = os.path.join(PATH,"yolov4-macaron1-labeling12.cfg")
dataPath    = os.path.join(PATH,"customData/obj.data")
namePath    = os.path.join(PATH,"customData/obj.names")
weightPath  = os.path.join(PATH,"backup/yolov4-macaron1-labeling12_4000.weights")

filter_size = 2 # 가운데 픽셀보다 얼마나 더 크게 볼건지 (정사각형 한 변)
xy_offset = 0.3 # 점 주위로 몇 m 안까지 하나의 객체로 볼건지 (군집화 할건지)

def array_msg(array):
    obs = PointCloud()
    # obs_clean = [[955920.0, 1950958.0],[955921.0, 1950958.0],[955919.0, 1950958.0],[955921.0, 1950959.0],[955922.0, 1950960.0],[955918.0, 1950958.0],[955920.0, 1950960.0]]
    
    for i in array:
        p = Point32()
        p.x = i[0]
        p.y = i[1]
        p.z = 0
        obs.points.append(p)

    return obs

def pick_xy(point_cloud, xy): # 차량 뒷 축 기준으로 해당 픽셀의 x z값 반환
    err, point_cloud_value = point_cloud.get_value(int(xy[0]), int(xy[1])) # zed sdk 설치해야 돌아감
    d = math.sqrt(point_cloud_value[2] ** 2 + point_cloud_value[0] ** 2 + point_cloud_value[1] ** 2)
    ratio = (0.25/3) * (d - 2.0) + 1.1 
    if ratio <= 1.0:
        ratio = 1.0
    #elif ratio >= 2.0:
    #    ratio = 2.0
    point_cloud_value[2] *= ratio
    point_cloud_value[0] *= ratio
    return point_cloud_value[2] + 0.71, -point_cloud_value[0]
    
def depth_filter1(point_cloud, pixel_xy): # pixel_xy [[x, y], [x, y]] 일케 넣어주면 왼쪽당, 오른쪽당 한번씩만 돌리면 됨
    filtered_xy = []
    # 인식된 라바콘 하나당
    for p_xy in pixel_xy:
        region_xy = [] # 주위 픽셀까지 뎁스 정보를 다 저장
        surrounding_dot_count = []# 주변 점 개수
            
        # 사각형으로 픽셀 뎁스정보 다 받아오기
        for i in range(filter_size):
            for j in range(filter_size):
                try: # 해상도를 넘어서 뎁스 가져오는거는 그냥 패스
                    x, y = pick_xy(point_cloud, [p_xy[0] - filter_size/2 + j, p_xy[1]  - filter_size/2 + i])
                    region_xy.append([x, y])
                except:
                    pass
            
        # 가장 주위에 많은 점 찾기
        for ind in range(len(region_xy)):
            dot_count = 0
            for every in region_xy:
                # 주위에 있는지
                if abs(region_xy[ind][0] - every[0]) < xy_offset and abs(region_xy[ind][1] - every[1]) < xy_offset:
                    dot_count += 1
            surrounding_dot_count.append(dot_count)

        temp = max(surrounding_dot_count)
        center_ind = surrounding_dot_count.index(temp)
        print(surrounding_dot_count, 'temp')

        x_temp = []
        y_temp = []
        for every in region_xy:
            # 주위에 있는것들 평균
            if abs(region_xy[center_ind][0] - every[0]) < xy_offset and abs(region_xy[center_ind][1] - every[1]) < xy_offset:
                x_temp.append(every[0])
                y_temp.append(every[1])

        filtered_xy.append([np.mean(x_temp), np.mean(y_temp)])

    return filtered_xy

def depth_filter2(point_cloud, pixel_xy): # pixel_xy [[x, y], [x, y]] 일케 넣어주면 왼쪽당, 오른쪽당 한번씩만 돌리면 됨
    filtered_xy = []
    # 인식된 라바콘 하나당
    for p_xy in pixel_xy:
        region_xy = [] # 주위 픽셀까지 뎁스 정보를 다 저장
        surrounding_dot_count = []# 주변 점 개수
        # 사각형으로 픽셀 뎁스정보 다 받아오기
        for i in range(filter_size):
            for j in range(filter_size):
                try: # 해상도를 넘어서 뎁스 가져오는거는 그냥 패스
                    x, y = pick_xy(point_cloud, [p_xy[0] - filter_size + 2 * j, p_xy[1]  - filter_size + 2 * i])
                    region_xy.append([x, y])
                except:
                    pass
        x_temp = []
        y_temp = []
        for i in region_xy:
            x_temp.append(i[0])
            y_temp.append(i[1])     
        mean_xy = ([np.mean(x_temp), np.mean(y_temp)])
        
        x_temp = []
        y_temp = []
        for every in region_xy:
            if abs(mean_xy[0] - every[0]) < xy_offset and abs(mean_xy[1] - every[1]) < xy_offset:
                x_temp.append(every[0])
                y_temp.append(every[1])

        filtered_xy.append([np.mean(x_temp), np.mean(y_temp)])

    return filtered_xy

def main():
    import rospy
    import array
    import time
    import os
    
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    # init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    # init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_minimum_distance = 1.0

    # Open the camera
    zed.open(init_params)
    
    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode

    runtime_parameters.confidence_threshold = 80
    runtime_parameters.textureness_confidence_threshold = 100

    #이미지 size 조정
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = 640
    image_size.height = 480

    rospy.init_node("object_dection")

    network, class_names, class_colors = darknet.load_network(
        configPath,
        dataPath,
        weightPath
    )


    blue_pub = rospy.Publisher('blue_rubber', PointCloud, queue_size=1)
    yellow_pub = rospy.Publisher('yellow_rubber', PointCloud, queue_size=1)
    # time_rec = time.time()

    while not rospy.is_shutdown():
        corn_blue = []
        corn_yellow = []
        # TEST END
        #현재 카메라 화면 읽어오기
        image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        #img에 현재 카메라가 담고있는 프레임 들어감
        err = zed.grab(runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS :
        # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            img = image_zed.get_data()

        point_cloud = sl.Mat()
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        
        image, detections = darknet_images.image_detection_cv2file(
            img, network, class_names, class_colors, .35
        )

        img_color = image
        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        lower_blue = (120-10, 30, 30)
        upper_blue = (120+10, 255,255)
        img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

        #detections는 이미지에서 탐지된 것들의 list
        #그 안에는 탐지된 classnum, draw box의 각 꼭짓점의 값
        #message type에 맞게 설정
        #여러개를 인식했을 수도 있기 때문에 list로 해서 list를 message의 각 값으로 함
        #detections = [[label,prob,[box,,..]],[label....]]
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
                left = left * 1920 / 416
                right = right * 1920 / 416
                top = top * 1080 / 416
                bottom = bottom *1080 / 416
                x_center = np.round((left+right)/2)
                y_center = np.round((top+bottom)/2)
                center_xy = [x_center, y_center]
                corn_blue.append(center_xy)
            elif blue_count*2 < count:
                cv2.putText(image, "{}".format('y'),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,0,255), 2)
                left = left * 1920 / 416
                right = right * 1920 / 416
                top = top * 1080 / 416
                bottom = bottom *1080 / 416
                x_center = np.round((left+right)/2)
                y_center = np.round((top+bottom)/2)
                center_xy = [x_center, y_center]
                corn_yellow.append(center_xy)
            else:
                cv2.putText(image, "{}".format('error'),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,0,255), 2)
                #416*416-->1280*720
        
        # time_rec2 = time.time()
        blue_corn_xyz = depth_filter2(point_cloud, corn_blue)
        yellow_corn_xyz = depth_filter2(point_cloud, corn_yellow)
        # blue_corn_xyz = [[10,20],[12,324]]
        # yellow_corn_xyz = [[130,240],[12,324]]
        
        # # obj_msg.data = [blue_corn_xyz, yellow_corn_xyz]
        # # obj_pub.publish(obj_msg)
        blue_msg = array_msg(blue_corn_xyz)
        blue_pub.publish(blue_msg)
        yellow_msg = array_msg(yellow_corn_xyz)
        yellow_pub.publish(yellow_msg)
        cv2.imshow('test', image)
        
        if cv2.waitKey(1) == ord('q'):
            break
        # print(time.time()-time_rec)
        # time_rec = time.time()
        # print((time.time() - time_rec2), 'depth')
        # time_rec2 = time.time()   
      
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
