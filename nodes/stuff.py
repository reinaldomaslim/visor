#!/usr/bin/env python

""" reinaldo
"""

import rospy
import numpy as np
import math
import cv2


import time

class Stuff(object):

    x, y=0, 0
    dist_thres=1
    dying_count=0
    dead=False

    def __init__(self, id_num, x, y):

        self.x=x
        self.y=y
        self.id_num=id_num

    def update(self, positions):
        if len(positions)==0:
            self.dying_count+=1
            if self.dying_count>2:
                self.dead=True

            return positions

        s=np.square(np.asarray(positions)-np.array([self.x, self.y]))
        rms=np.sum(s, axis=1)
        min_arg=np.argmin(rms)
        
        if rms[min_arg]<self.dist_thres:
            # self.my_pos[0]=positions[min_arg]
            self.x=positions[min_arg][0]
            self.y=positions[min_arg][1]

            self.dying_count=0
            positions=np.delete(positions, min_arg, 0)
        else:
            self.dying_count+=1

        if self.dying_count>2:
            self.dead=True  


        return positions


# a=Stuff(0, 0, 0)
# positions=[[5, 4], [8, 3], [0.1, 0.2], [0.3, 0.6]]
# positions=a.update(positions)
# print(a.my_pos)

# print(positions)

