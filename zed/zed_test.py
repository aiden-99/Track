#!/usr/bin/env python3
# -- coding: utf-8 --
########################################################################
#
# Copyright (c) 2021, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
import math
import rospy
import numpy as np
import sys

from std_msgs.msg import Float64MultiArray


class zed():
    def __init__(self):
        self.xyz_pub = rospy.Publisher("xyz", Float64MultiArray, queue_size=1)

    def main(self):
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
        # Setting the depth confidence parameters
        runtime_parameters.confidence_threshold = 100
        runtime_parameters.textureness_confidence_threshold = 100

        # Capture 150 images and depth, then stop
        i = 0
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()

        mirror_ref = sl.Transform()
        mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
        tr_np = mirror_ref.m

        while i < 150:
            # A new image is available if grab() returns SUCCESS
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve left image
                zed.retrieve_image(image, sl.VIEW.LEFT)
                # Retrieve depth map. Depth is aligned on the left image
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                # Retrieve colored point cloud. Point cloud is aligned on the left image.
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

                # Get and print distance value in mm at the center of the image
                # We measure the distance camera - object using Euclidean distance
                x = round(image.get_width() / 2) - 100
                y = round(image.get_height() / 2)
                err, point_cloud_value = point_cloud.get_value(x, y)

                distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                    point_cloud_value[1] * point_cloud_value[1] +
                                    point_cloud_value[2] * point_cloud_value[2])

                point_cloud_np = point_cloud.get_data()
                point_cloud_np.dot(tr_np)

                if not np.isnan(distance) and not np.isinf(distance):
                    # print("Distance to Camera at ({}, {}) (image center): {:1.3} m".format(x, y, distance), end="\r")
                    print("x, y, z", point_cloud_value[0], point_cloud_value[1], point_cloud_value[2], end="\r")
                    point = Float64MultiArray()
                    point.data = [point_cloud_value[0], point_cloud_value[1], point_cloud_value[2]]

                    self.xyz_pub.publish(point)

                    # Increment the loop
                    i = i + 1
                else:
                    print("Can't estimate distance at this position.")
                    print("Your camera is probably too close to the scene, please move it backwards.\n")
                sys.stdout.flush()

        # Close the camera
        zed.close()

if __name__ == "__main__":
    rospy.init_node("zed_test", anonymous=True)
    z = zed()
    z.main()


'''
import rospy
from sensor_msgs.msg import PointCloud2
# import pyzed.sl as sl


class sub_zed:
    def __init__(self):
        self.pose_sub = rospy.Subscriber('/zed2/zed_node/point_cloud/cloud_registered', PointCloud2, self.PointCloud_callback, queue_size = 10)
        self.x = 0
        self.y = 0
        self.z = 0
        self.color = 0


    def PointCloud_callback(self, point_cloud):
        point3D = point_cloud.data
        # self.x = point3D[0]
        # self.y = point3D[1]
        # self.z = point3D[2]
        # self.color = point3D[3]
        print(len(point3D))

    def print_data(self):
        print(self.x, self.y, self.z, self.color)


def main():
    rate = rospy.Rate(5)
    zed = sub_zed()
    while not rospy.is_shutdown():
        zed.print_data()
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node("zed_test", anonymous=True)
    main()
    '''