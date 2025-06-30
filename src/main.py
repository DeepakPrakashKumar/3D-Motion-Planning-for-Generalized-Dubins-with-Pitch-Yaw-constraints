# This is the main function to be called for generating a feasible solution connecting the initial and final configurations
# by connecting osculating spheres with same directionality by a cylindrical envelope, and osculating spheres with different
# directionality by conical envelopes.

import os
import sys
from main_functions_heuristic import generate_random_configs_3D, Dubins_3D_numerical_path_on_surfaces
from pathlib import Path
import math
import numpy as np
from math import cos as cos
from math import sin as sin

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

## OPTION 1: PROVIDE INITIAL AND FINAL CONFIGURATIONS DIRECTLY
# Here, we provide the description of the initial and final configurations if we
# want to provide the initial and final configurations. For this implementation, we instead generate random initial and final configurations.
# ini_config = np.array([[0, 0, 0],\
#                        [1, 0, 0],\
#                        [0, 1, 0],\
#                        [0, 0, 1]])

# fin_config = np.array([[30, 10, 15],\
#                        [1/math.sqrt(2), -1/math.sqrt(2), 0],\
#                        [1/math.sqrt(2), 1/math.sqrt(2), 0],\
#                        [0, 0, 1]])

## OPTION 2: PROVIDE INITIAL AND FINAL LOCATION, HEADING, PITCH, AND ROLL ANGLES
# Alternatively, we provide the initial and final location and heading, pitch, and roll angles
# ini_loc = np.array([0, 0, 0])
# ini_heading_angle = 30*math.pi/180
# ini_pitch_angle = 10*math.pi/180 # HERE, PITCH ANGLE IS MEASURED WITH RESPECT TO -Y AXIS, SINCE WE ARE DEFINING PITCH ANGLE AS THE
# # ANGLE MADE WITH RESPECT TO THE XY PLANE, WHEREIN PITCH ANGLE IS POSITIVE WHEN THE AIRCRAFT IS PITCHED UPWARD
# ini_roll_angle = 15*math.pi/180

# fin_loc = np.array([5, 10, 15])
# fin_heading_angle = 190*math.pi/180
# fin_pitch_angle = 10*math.pi/180
# fin_roll_angle = -15*math.pi/180

ini_loc = np.array([0, 0, 0])
ini_heading_angle = -30*math.pi/180
ini_pitch_angle = 10*math.pi/180 # HERE, PITCH ANGLE IS MEASURED WITH RESPECT TO -Y AXIS, SINCE WE ARE DEFINING PITCH ANGLE AS THE
# ANGLE MADE WITH RESPECT TO THE XY PLANE, WHEREIN PITCH ANGLE IS POSITIVE WHEN THE AIRCRAFT IS PITCHED UPWARD
ini_roll_angle = 15*math.pi/180

fin_loc = np.array([0, -30, 5])
fin_heading_angle = 190*math.pi/180
fin_pitch_angle = 10*math.pi/180
fin_roll_angle = -15*math.pi/180

# Obtaining the tangent vector, tangent normal vector, and surface normal vector
ini_tang = np.array([cos(ini_heading_angle)*cos(ini_pitch_angle), sin(ini_heading_angle)*cos(ini_pitch_angle), sin(ini_pitch_angle)])
fin_tang = np.array([cos(fin_heading_angle)*cos(fin_pitch_angle), sin(fin_heading_angle)*cos(fin_pitch_angle), sin(fin_pitch_angle)])
# Computing the tangent normal vector
ini_heading_norm = cos(ini_roll_angle)*np.array([cos(ini_heading_angle + math.pi/2), sin(ini_heading_angle + math.pi/2), 0])\
        + math.sin(ini_roll_angle)*np.cross(ini_tang, np.array([cos(ini_heading_angle + math.pi/2), sin(ini_heading_angle + math.pi/2), 0]))
fin_heading_norm = cos(fin_roll_angle*math.pi/180)*np.array([cos(fin_heading_angle + math.pi/2), sin(fin_heading_angle + math.pi/2), 0])\
        + math.sin(fin_roll_angle*math.pi/180)*np.cross(fin_tang, np.array([cos(fin_heading_angle + math.pi/2), sin(fin_heading_angle + math.pi/2), 0]))

# Computing the surface normal vector
ini_norm = np.cross(ini_tang, ini_heading_norm)
fin_norm = np.cross(fin_tang, fin_heading_norm)

ini_config = np.array([ini_loc, ini_tang, ini_heading_norm, ini_norm])
fin_config = np.array([fin_loc, fin_tang, fin_heading_norm, fin_norm])

## OPTION 3: RANDOM INITIAL AND FINAL CONFIGURATIONS
# If a random initial and final configuration ought to be generated, we randomly generate the initial and final configurations.
xlim = 40; ylim = 40; zlim = 40; # We provide the region in the 3d space wherein we want to generate the configuration.
# xlim is used to pick a random x coordinate between 0 and xlim; similar interpretation follows for ylim and zlim.
# ini_config = generate_random_configs_3D(xlim, ylim, zlim)
# fin_config = generate_random_configs_3D(xlim, ylim, zlim)

xgrid_size = [-xlim/2, xlim/2]
ygrid_size = [-ylim/2, ylim/2]
zgrid_size = [-zlim/2, zlim/2]

# xgrid_size = [-5, 30]
# ygrid_size = [-5, 30]
# zgrid_size = [-5, 30]

# Providing additional parameters for the plot function
length_vec_orientation = 15 # Length of the vectors in the plot
scale_aircraft = 5 # Scale of the aircraft in the plot
# elev = 23 # Elevation angle for the plot (for viewing)
# azimuth = -135 # Azimuth angle for the plot (for viewing)
# elev = 20 # Elevation angle for the plot (for viewing)
# azimuth = -158 # Azimuth angle for the plot (for viewing)
elev = 16 # Elevation angle for the plot (for viewing)
azimuth = -8 # Azimuth angle for the plot (for viewing)

# We now provide the parameters for the vehicle
# R = 10 # Radius of the osculating sphere
# r = 5 # This is the radius of the tight turn for the vehicle.
# pitch_rate = 0.1 # In radians/sec
# yaw_rate = 0.15 # In radians/sec

# We now obtain the radius of the spheres corresponding to the pitch rate and yaw rate
R_pitch = 40
R_yaw = 50

# We also obtain the minimum turning radius for when both pitch and yaw rates are attaining its maximum
# absolute value; these turns occur on spheres
# r_min = 1/math.sqrt(pitch_rate**2 + yaw_rate**2)
r_min = 1/math.sqrt((1/R_pitch)**2 + (1/R_yaw)**2)

# We provide the number of discretizations to be considered for the location and the heading angle
disc_no_loc = 15
disc_no_heading = 15

# We call the main function that constructs feasible solutions through sphere-cylinder-sphere, sphere-plane-sphere, and sphere-sphere-sphere
# combinations, and returns the path and orientations corresponding to the best path.
min_dist_path_length, min_dist_path_pts, tang_global_path, tang_normal_global_path, surf_normal_global_path, path_type =\
      Dubins_3D_numerical_path_on_surfaces(ini_config, fin_config, r_min, R_yaw, R_pitch, disc_no_loc, disc_no_heading,\
                                            visualization = 1, vis_best_surf_path = 1, vis_int = 0, filename = 'temp.html')

print('The path type is ' + path_type + '. The minimum distance is', min_dist_path_length, '.')

# We now simulate the motion of a vehicle along the path that we have obtained.
plot_trajectory(ini_config, fin_config, min_dist_path_pts, tang_global_path, tang_normal_global_path, surf_normal_global_path, path_type, R_yaw, R_pitch,\
                 xgrid_size = xgrid_size, ygrid_size = ygrid_size, zgrid_size = zgrid_size, length_vec_orientation = length_vec_orientation, scale_aircraft = scale_aircraft,\
                  elev = elev, azim = azimuth, animate = False, int_config_spacing = 45, video_name = False)