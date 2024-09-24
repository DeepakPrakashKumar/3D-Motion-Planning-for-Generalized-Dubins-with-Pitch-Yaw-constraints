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

from Plane_Dubins_functions import *

# Declaring the parameters
r = 1

ini_config = np.array([0, 0, 0])
fin_config = np.array([0, 0, math.pi])

path_lengths, opt_path_type_configs, x, y = optimal_dubins_path(ini_config, fin_config, r)