# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 21:40:53 2022

@author: deepa
"""

import os

path = 'D:\TAMU\Research\Cylinder and sphere'

os.chdir(path)
from cylinder_sphere_functions import *

# Setting the parameters
# Setting the limits for the generation of the random initial and final configurations
xlim = 40
ylim = 40
zlim = 40
r = 5
R = 10 # Radius of the sphere and cylinder
disc_no = 3 # number of discretizations

# Generating the random configurations
ini_config = generate_random_configs_3D(xlim, ylim, zlim)
fin_config = generate_random_configs_3D(xlim, ylim, zlim)

# _, _, _, sp1, cyc, sp2 = Dubins_3D_numerical_path_on_surfaces(ini_config, fin_config, r, R, disc_no)
Dubins_3D_numerical_path_on_surfaces(ini_config, fin_config, r, R, disc_no)