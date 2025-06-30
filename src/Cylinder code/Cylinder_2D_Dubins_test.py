# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 15:16:03 2021

@author: Deepak Prakash Kumar
"""

import numpy as np
import math
import os

from Cylinder_2D_Dubins_functions_simple import *
import time

# Radius of the cylinder
R = 10
# Axis of the cylinder (assumed)
axis = np.array([0, 0, 1])
# Maximum z-coordinate of randomly generated points
zmax = 20
rad_tight_turn = 5

ini_pos, ini_tang_vect, final_pos, final_tang_vect = generate_random_configs_cylinder(R, zmax)

start_time = time.time()
length, path_type, points_path, tangents_path, normal_path = generate_visualize_path(ini_pos, ini_tang_vect, R, final_pos, final_tang_vect, zmax, 0, rad_tight_turn, path_config = 1, filename = False)
print('Time taken is ', time.time() - start_time, '. Start time is ', start_time, ' and end time is ', time.time())
print('Optimal path is of type ', path_type, ' with length ', length, ' and parameters ', points_path, tangents_path, normal_path)