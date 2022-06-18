#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing_extensions import runtime
import darknet
import darknet_images
import cv2
import os, sys
import pyzed.sl as sl
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from sensor.webcam import webcam
from macaron_2.msg import Traffic

#경로설정
PATH = os.path.abspath(os.path.dirname(__file__))
#configPath  = os.path.join(PATH,"919data/yolo-obj.cfg")
configPath  = os.path.join(PATH,"fin/yolo-obj.cfg")
dataPath    = os.path.join(PATH,"fin/obj.data")
namePath    = os.path.join(PATH,"fin/obj.names")
#namePath    = os.path.join(PATH,"obj2.names")
#weightPath  = os.path.join(PATH,"919data/yolo-obj_last2.weights")
#weightPath  = os.path.join(PATH,"yolo-obj_2000.weights")
weightPath  = os.path.join(PATH,"fin/yolo-obj_15000.weights")

def main():
    import rospy
    import array
    import time
    import os
    
    init = sl.InitParameters()
    init.coordinate_units = sl.UNIT.UNIT_METER
    #cam = webcam(2)
    cam = sl.Camera()
    status = cam.open(init)
    if(status != sl.ERROR_CODE.SUCCESS):
        print("zed camera is not opened!!!")
        exit(-1)

    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD
    mat = sl.Mat()
    point_cloud_mat = sl.Mat()

    rospy.init_node("object_dection")
    rate = rospy.Rate(2) 
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

    while not rospy.is_shutdown():
        # FOR TEST
        
        """
        i+=1
        print(base_path + imagePath[i])
        img = cv2.imread(base_path + imagePath[i])
        """
        # TEST END
        #현재 카메라 화면 읽어오기
        #img = cam.capture(2)
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.VIEW_LEFT)
            img = mat.get_data()

        #이 밑에 두줄은 안해도 되는 듯
        cam.retrieve_measure(point_cloud_mat, sl.MEASURE_XYZRGBA)
        depth = point_cloud_mat.get_data()


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
            classes.append(label)
            #classes += label
            xmin.append(int(box[0]))
            ymin.append(int(box[1]))
            xmax.append(int(box[2]))
            ymax.append(int(box[3]))
            print("result :", label, prob, box)
        #image = ObjectDetection.draw_boxes(img, detection)
        #print("i, length", i, len(detections))
        obj_msg.Class = classes
        obj_msg.xmin = xmin
        obj_msg.ymin = ymin
        obj_msg.xmax = xmax
        obj_msg.ymax = ymax

        obj_pub.publish(obj_msg)


        #filename = '/home/macaron/test_run/' + str(cur_time) + '/test' + str(i) + '.jpg' # 
        #filename2 = '/home/macaron/test_run/' + str(cur_time) + '/detected' + str(i) + '.jpg' # 
        #i += 1
        #cv2.imwrite(filename, image) 
        #cv2.imwrite(filename2, image2) 
        #cv2.imshow("result", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        rate.sleep()
    

if __name__ == "__main__":
    main()
