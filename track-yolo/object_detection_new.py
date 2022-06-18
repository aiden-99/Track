#!/usr/bin/env python
# -*- coding: utf-8 -*-
import darknet
import darknet_images
import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import webcam
from std_msgs.msg import Int16MultiArray

PATH = os.path.abspath(os.path.dirname(__file__))
#configPath  = os.path.join(PATH,"919data/yolo-obj.cfg")
configPath  = os.path.join(PATH,"yolov4-macaron1-labeling12.cfg")
dataPath    = os.path.join(PATH,"customData/obj.data")
namePath    = os.path.join(PATH,"customData/obj.names")
#namePath    = os.path.join(PATH,"obj2.names")
#weightPath  = os.path.join(PATH,"919data/yolo-obj_last2.weights")
#weightPath  = os.path.join(PATH,"yolo-obj_2000.weights")
weightPath  = os.path.join(PATH,"backup/yolov4-macaron1-labeling12_10000.weights")

def main():
    import rospy
    import array
    import time
    import os
    #cam = webcam()
    rospy.init_node("object_dection")
    rate = rospy.Rate(2) 
    network, class_names, class_colors = darknet.load_network(
        configPath,
        dataPath,
        weightPath
    )

    # TEST END
    #cur_time = time.time()
    #os.mkdir('/home/macaron/test_run/' + str(cur_time))
    #i = 0
    #testimg = ['','2','75','160','324','371','391','404','429','545','547']
    #testimg = range(1,19)
    #print(testimg)
    while not rospy.is_shutdown():
        # FOR TEST
        
        """
        i+=1
        print(base_path + imagePath[i])
        img = cv2.imread(base_path + imagePath[i])
        """
        obj_pub = rospy.Publisher('traffic_obj', Int16MultiArray, queue_size=1)
        #Traffic이라는 메시지 타입의 객체를 obj_msg라는 이름으로 생성
        obj_msg = Int16MultiArray()
        # TEST END
        #현재 카메라 화면 읽어오기
        #img = cam.capture(2)
        img = cv2.imread('1.jpg')
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
        darknet.draw_boxes(detections, image, class_colors)
        
        #indexing 리스트 안에 탐지된 객체의 인덱스를 삽입
        classes = ['traffic_green', 'traffic_left_green', 'traffic_all_green', 'traffic_orange', 'traffic_red', 
                   'delivery_a1', 'delivery_a2', 'delivery_a3', 'delivery_b1', 'delivery_b2', 'delivery_b3']
        
        for label, prob, box in detections:
            obj_msg.data.append(classes.index(label))

        obj_pub.publish(obj_msg)

        #filename = '/home/macaron/test_run/' + str(cur_time) + '/test' + str(i) + '.jpg' # 
        #filename2 = '/home/macaron/test_run/' + str(cur_time) + '/detected' + str(i) + '.jpg' # 
        #i += 1
        #cv2.imwrite(filename, image) 
        #cv2.imwrite(filename2, image2) 
        #cv2.imshow("result", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #rate.sleep()
    

if __name__ == "__main__":
    main()
