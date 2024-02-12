# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 15:16:03 2021

@author: deepa
"""

import numpy as np
import math
import os

path = 'D:\TAMU\Research\Cylinder code'

os.chdir(path)

# from Cylinder_2D_Dubins_functions_simple import generate_visualize_path, unwrapped_configurations_2D
# from Cylinder_2D_Dubins_functions import generate_random_configs_cylinder, generate_visualize_path_simple,\
#     transformation_point_2D
# from Cylinder_2D_Dubins_functions_simple import *
import Cylinder_2D_Dubins_functions as Cf
# import Cylinder_2D_Dubins_functions_simple as Cfs

# Radius of the cylinder
R = 10
# Axis of the cylinder (assumed)
axis = np.array([0, 0, 1])
# Maximum z-coordinate of randomly generated points
zmax = 20
rad_tight_turn = 5

ini_pos, ini_tang_vect, final_pos, final_tang_vect = Cf.generate_random_configs_cylinder(R, zmax)
# transformation_point_2D(ini_pos, ini_tang_vect, R, final_pos, final_tang_vect, 1, 20)
# generate_visualize_path(ini_pos, ini_tang_vect, R, final_pos, final_tang_vect, zmax, 2, rad_tight_turn, 'test_old.html')
Cf.generate_visualize_path_simple(ini_pos, ini_tang_vect, R, final_pos, final_tang_vect, zmax, 2, rad_tight_turn, 'test_new.html')
# generate_visualize_path(ini_pos, ini_tang_vect, R, final_pos, final_tang_vect, zmax, 1, rad_tight_turn, 'test.html')

# path = 'D:\TAMU\Research\Cylinder code'

# os.chdir(path)

# Cfs.generate_visualize_path(ini_pos, ini_tang_vect, R, final_pos, final_tang_vect, zmax, 1, rad_tight_turn, 'test.html')

#%%

import numpy as np
import math
import os

path = 'D:\TAMU\Research\Cylinder code'

os.chdir(path)
import Cylinder_2D_Dubins_functions_simple as Cfs

# Radius of the cylinder
R = 10
# Axis of the cylinder (assumed)
axis = np.array([0, 0, 1])
# Maximum z-coordinate of randomly generated points
zmax = 20
rad_tight_turn = 5

ini_pos, ini_tang_vect, final_pos, final_tang_vect = Cfs.generate_random_configs_cylinder(R, zmax)
Cfs.generate_visualize_path(ini_pos, ini_tang_vect, R, final_pos, final_tang_vect, zmax, 1, rad_tight_turn, 'test.html')