# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:07:36 2022

@author: deepa
"""

import numpy as np
import math
import os
import pandas as pd
import sys
import time

from Plane_Dubins_functions import optimal_dubins_path

# Declaring the parameters
r = 1

ini_config = np.array([0, 0, 0])
fin_config = np.array([0, 0, math.pi])

for counter in range(10):
    start_time = time.time()
    path_length, path_params_opt, opt_path_type_configs, x, y, heading = optimal_dubins_path(ini_config, fin_config, r, path_config = 0, filename = False)
    print('Time taken is ', time.time() - start_time)

# print('Optimal path is of type ', opt_path_type_configs, ' with length ', path_length, ' and parameters ', path_params_opt)