# This is the main function to be called for generating a feasible solution connecting the initial and final configurations
# by connecting osculating spheres with same directionality by a cylindrical envelope, and osculating spheres with different
# directionality by conical envelopes.

import os
import sys
from main_functions_heuristic import generate_random_configs_3D, Dubins_3D_numerical_path_on_surfaces
from pathlib import Path
import math
import numpy as np

# Including the following command to ensure that python is able to find the relevant files afer changing directory
sys.path.insert(0, '')
# Obtaining the current directory
cwd = os.getcwd()
current_directory = Path(__file__).parent
path_str = str(current_directory)

# Importing code for plotting
rel_path = '\Visualization'
os.chdir(path_str + rel_path)
from visualization_simulation import plot_trajectory

# Returning to initial directory
os.chdir(cwd)

# Here, we provide the description of the initial and final configurations if we
# want to provide the initial and final configurations. For this implementation, we instead generate random initial and final configurations.
ini_config = np.array([[0, 0, 0],\
                       [1, 0, 0],\
                       [0, 1, 0],\
                       [0, 0, 1]])

fin_config = np.array([[30, 10, 15],\
                       [1/math.sqrt(2), -1/math.sqrt(2), 0],\
                       [1/math.sqrt(2), 1/math.sqrt(2), 0],\
                       [0, 0, 1]])

# If a random initial and final configuration ought to be generated, we randomly generate the initial and final configurations.
xlim = 40; ylim = 40; zlim = 40; # We provide the region in the 3d space wherein we want to generate the configuration.
# xlim is used to pick a random x coordinate between 0 and xlim; similar interpretation follows for ylim and zlim.
ini_config = generate_random_configs_3D(xlim, ylim, zlim)
fin_config = generate_random_configs_3D(xlim, ylim, zlim)

xgrid_size = [-xlim/2, xlim/2]
ygrid_size = [-ylim/2, ylim/2]
zgrid_size = [-zlim/2, zlim/2]

# xgrid_size = [-5, 30]
# ygrid_size = [-5, 30]
# zgrid_size = [-5, 30]

length_vec_orientation = 5
scale_aircraft = 2
elev = 23
azimuth = -135

# We now provide the parameters for the vehicle
# R = 10 # Radius of the osculating sphere
# r = 5 # This is the radius of the tight turn for the vehicle.
pitch_rate = 0.1 # In radians/sec
yaw_rate = 0.15 # In radians/sec

# We now obtain the radius of the spheres corresponding to the pitch rate and yaw rate
R_pitch = 1/pitch_rate
R_yaw = 1/yaw_rate

# We also obtain the minimum turning radius for when both pitch and yaw rates are attaining its maximum
# absolute value; these turns occur on spheres
r_min = 1/math.sqrt(pitch_rate**2 + yaw_rate**2)

# We provide the number of discretizations to be considered for the location and the heading angle
disc_no = 5

# We call the main function that constructs feasible solutions through sphere-cylinder-sphere, sphere-plane-sphere, and sphere-sphere-sphere
# combinations, and returns the path and orientations corresponding to the best path.
min_dist_path_length, min_dist_path_pts, tang_global_path, tang_normal_global_path, surf_normal_global_path, path_type =\
      Dubins_3D_numerical_path_on_surfaces(ini_config, fin_config, r_min, R_yaw, R_pitch, disc_no, visualization = 1, filename = 'temp.html')

# print('Tangent vectors for the path are:\n', tang_global_path)

if "left" in path_type or "right" in path_type:
    R = R_yaw
else:
    R = R_pitch

# We now simulate the motion of a vehicle along the path that we have obtained.
# plot_trajectory(ini_config, fin_config, min_dist_path_pts, tang_global_path, tang_normal_global_path, surf_normal_global_path, path_type, R,\
#                  xgrid_size = xgrid_size, ygrid_size = ygrid_size, zgrid_size = zgrid_size, length_vec_orientation = length_vec_orientation, scale_aircraft = scale_aircraft,\
#                   elev = elev, azim = azimuth, video_name = 'cross_tangent.mp4')