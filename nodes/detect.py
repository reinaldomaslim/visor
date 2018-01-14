#!/usr/bin/env python

""" reinaldo
"""

import rospy
import numpy as np
import math
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseStamped
from sensor_msgs.msg import RegionOfInterest, CameraInfo, LaserScan, Image
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from visualization_msgs.msg import Marker
from ssd import ssd_detect, graph

import time

class VisorDetect(object):

    fov_x=57
    fov_y=42
    height=0.86
    pitch=0 

    skip=1

    count=0

    def __init__(self):
        rospy.init_node("visor_detect", anonymous="False")

        #subscribe to camera image
        # rospy.Subscriber("/usb_cam/image_raw", Image, self.img_callback, queue_size = 1)
        rospy.Subscriber("/camera_img", Image, self.img_callback, queue_size = 1)
        self.bridge = CvBridge()

        self.img_pub=rospy.Publisher('/detect_img', Image, queue_size=1)

        rate=rospy.Rate(10)

        while not rospy.is_shutdown():
            rate.sleep()


    def img_callback(self, msg):
        if self.count%self.skip==0:
            start=time.time()
            img=self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # img=self.img_correction(img)

            with graph.as_default():

                r = ssd_detect(img)
                # print(r[0])
                # print(r[2])

            result=img.copy()
            for rect in r[2]:
                y_min=int(rect[0]*img.shape[0])
                y_max=int(rect[2]*img.shape[0])
                x_min=int(rect[1]*img.shape[1])
                x_max=int(rect[3]*img.shape[1])
                result=cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
                
                depth, lat=self.get_pos((x_min+x_max)/2, y_max, img.shape)

                result=cv2.putText(result,str(depth)+","+str(lat), (x_min+10, y_min+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2,cv2.LINE_AA)


            self.img_pub.publish(self.bridge.cv2_to_imgmsg(result, "bgr8"))
            

            print(str(time.time()-start))
            self.count=0
        else:
            pass

        self.count+=1


    def get_pos(self, x_avg, y_max, img_shape):
        
        h, w=img_shape[0], img_shape[1] 

        ratio_y=(float(y_max)-h/2)/(h/2)
        ratio_angle_y=math.atan(ratio_y*math.tan(0.5*self.fov_y*math.pi/180))/(0.5*self.fov_y*math.pi/180)
        beta=(ratio_angle_y*self.fov_y/2+self.pitch+0.001)*math.pi/180
        depth=self.height/math.tan(beta)

        ratio_x=(float(x_avg)-w/2)/(w/2)
        ratio_angle_x=math.atan(ratio_x*math.tan(0.5*self.fov_x*math.pi/180))/(0.5*self.fov_x*math.pi/180)
        alpha=(ratio_angle_x*self.fov_x/2+0.001)*math.pi/180
        lateral=-depth*math.tan(alpha)


        return round(depth, 3), round(lateral, 3)


    def img_correction(self, img):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
        res=np.zeros_like(img)
        for i in range(3):
            res[:, :, i] = clahe.apply(img[:, :, i])
        return res




##########################
##########main############
##########################


if __name__ == '__main__':

    try:
        VisorDetect()
    except rospy.ROSInterruptException:
        rospy.loginfo("finished.")