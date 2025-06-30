import numpy as np
import math
from math import cos as cos
from math import sin as sin
from main_functions_heuristic import Dubins_3D_numerical_path_on_surfaces
import time
import pickle
import csv

# We define the instances to be tested, which were considered from "Minimal 3D Dubins Path with Bounded Curvature and Pitch Angle"
deg2rad = lambda x : math.pi * x / 180.

DUMMY = [
  [
    [0, 0, 0, deg2rad(0), deg2rad(0)],
    [100, 0, 0, deg2rad(0), deg2rad(0)],
    [0, 0],
    "Dummy"
  ]
] # Dummy instance is used for warm starting the processes used for parallelization

LONG = [
  [
    [200, 500, 200, deg2rad(180), deg2rad(-5)],
    [500, 350, 100, deg2rad(0), deg2rad(-5)],
    [467.70, 449.56],
    "Long 1"
  ],[
    [100, -400, 100, deg2rad(30), deg2rad(0)],
    [500, -700, 0, deg2rad(150), deg2rad(0)],
    [649.52, 636.12],
    "Long 2"
  ],[
    [-200, 200, 250, deg2rad(240), deg2rad(15)],
    [ 500, 800,   0, deg2rad(45), deg2rad(15)],
    [1088.10, 1063.41],
    "Long 3"
  ],[
    [-300, 1200, 350, deg2rad(160), deg2rad(0)],
    [1000,  200,   0, deg2rad(30), deg2rad(0)],
    [1802.60, 1789.21],
    "Long 4"
  ],[
    [-500, -300, 600, deg2rad(150), deg2rad(10)],
    [1200,  900, 100, deg2rad(300), deg2rad(10)],
    [2245.14, 2216.40],
    "Long 5"
  ]
]

SHORT = [
  [
    [120, -30, 250, deg2rad(100), deg2rad(-10)],
    [220, 150, 100, deg2rad(300), deg2rad(-10)],
    [588.60, 583.47],
    "Short 1"
  ],[
    [380, 230, 200, deg2rad(30), deg2rad(0)],
    [280, 150,  30, deg2rad(200), deg2rad(0)],
    [667.71, 658.53],
    "Short 2"
  ],[
    [-80, 10, 250, deg2rad(20), deg2rad(0)],
    [ 50, 70,   0, deg2rad(240), deg2rad(0)],
    [979.34, 968.25],
    "Short 3"
  ],[
    [400, -250, 600, deg2rad(350), deg2rad(0)],
    [600, -150, 300, deg2rad(150), deg2rad(0)],
    [1169.73, 1161.55],
    "Short 4"
  ],[
    [-200, -200, 450, deg2rad(340), deg2rad(0)],
    [-300,  -80, 100, deg2rad(100), deg2rad(0)],
    [1367.56, 1354.12],
    "Short 5"
  ]
]

ADDITIONAL = [
    [
        [120, 40, 20, deg2rad(90), deg2rad(-5)],
        [300, 40, 15, deg2rad(-90), deg2rad(-5)],
        [],
        "Additional 1"
    ],
    [
        [120, 40, 20, deg2rad(90), deg2rad(-15)],
        [130, 120, 41, deg2rad(85), deg2rad(20)],
        [],
        "Additional 2"
    ]
]

# LONG = [
#   [
#     [200, 500, 200, deg2rad(180), deg2rad(-5)],
#     [500, 350, 100, deg2rad(0), deg2rad(-5)],
#     [467.70, 449.56],
#     "Long 1"
#   ]
# ]

# SHORT = [
# ]

# LONG = [
# ]

# SHORT = [
#   [
#     [400, -250, 600, deg2rad(350), deg2rad(0)],
#     [600, -150, 300, deg2rad(150), deg2rad(0)],
#     [1169.73, 1161.55],
#     "Short 4"
#   ]
# ]

# We define additional parameters/variables
Rpitch = 40
Ryaw = [30, 40, 50]
roll_angle_arr = [-15, 0, 15] # In degrees

roll_angle_arr_comb = [[roll_angle_arr[0], roll_angle_arr[1]], [roll_angle_arr[1], roll_angle_arr[2]], [roll_angle_arr[2], roll_angle_arr[0]]]

# We also define the number of discretizations considered
disc_no_loc = 15
disc_no_heading = 15

for data in [DUMMY, LONG, SHORT, ADDITIONAL]:
    for i in range(len(data)):
    #     for j in range(len(roll_angle_arr)):
        for roll_angle_arr_val in roll_angle_arr_comb:
        # for k in range(len(roll_angle_arr)):

            # We construct the initial and final configuration
            ini_loc = data[i][0][:3]
            fin_loc = data[i][1][:3]

            # Obtaining the heading angle and pitch angle
            ini_heading_angle = data[i][0][3]; fin_heading_angle = data[i][1][3]
            ini_pitch_angle = data[i][0][4]; fin_pitch_angle = data[i][1][4]

            # Obtaining the initial and final roll angles
            # ini_roll_angle = deg2rad(roll_angle_arr[j]); fin_roll_angle = deg2rad(roll_angle_arr[k])
            ini_roll_angle = deg2rad(roll_angle_arr_val[0]); fin_roll_angle = deg2rad(roll_angle_arr_val[1])

            # We now compute the orientation vectors
            ini_tang = np.array([cos(ini_heading_angle)*cos(ini_pitch_angle), sin(ini_heading_angle)*cos(ini_pitch_angle), sin(ini_pitch_angle)])
            fin_tang = np.array([cos(fin_heading_angle)*cos(fin_pitch_angle), sin(fin_heading_angle)*cos(fin_pitch_angle), sin(fin_pitch_angle)])

            ini_tang_norm = cos(ini_roll_angle)*np.array([cos(ini_heading_angle + math.pi/2), sin(ini_heading_angle + math.pi/2), 0])\
                    + math.sin(ini_roll_angle)*np.cross(ini_tang, np.array([cos(ini_heading_angle + math.pi/2), sin(ini_heading_angle + math.pi/2), 0]))
            fin_tang_norm = cos(fin_roll_angle)*np.array([cos(fin_heading_angle + math.pi/2), sin(fin_heading_angle + math.pi/2), 0])\
                    + math.sin(fin_roll_angle)*np.cross(fin_tang, np.array([cos(fin_heading_angle + math.pi/2), sin(fin_heading_angle + math.pi/2), 0]))
            
            ini_norm = np.cross(ini_tang, ini_tang_norm)
            fin_norm = np.cross(fin_tang, fin_tang_norm)

            # We now compute the initial and final configurations
            ini_config = np.array([ini_loc, ini_tang, ini_tang_norm, ini_norm])
            fin_config = np.array([fin_loc, fin_tang, fin_tang_norm, fin_norm])

            # print('Initial location', ini_loc, 'Final location', fin_loc, 'Initial heading angle', ini_heading_angle, 'Final heading angle', fin_heading_angle,\
            #     'Initial pitch angle', ini_pitch_angle, 'Final pitch angle', fin_pitch_angle)

            # Finally, we run through the variations in the turning radius
            for l in range(len(Ryaw)):

                # We now compute the minimum turning radius
                r_min = 1/math.sqrt((1/Rpitch)**2 + (1/Ryaw[l])**2)

                start_time = time.time()
                min_dist_path_length, min_dist_path_pts, tang_global_path, tang_normal_global_path, surf_normal_global_path, path_type =\
                    Dubins_3D_numerical_path_on_surfaces(ini_config, fin_config, r_min, Ryaw[l], Rpitch, disc_no_loc, disc_no_heading, visualization = 1,\
                                                            vis_best_surf_path = 1, filename = data[i][3] + ' ' + 'ini_roll ' + str(roll_angle_arr_val[0])\
                                                                + ' fin_roll ' + str(roll_angle_arr_val[1]) + ' Ryaw ' + str(Ryaw[l]) + '.html')
                runtime = time.time() - start_time

                print('Instance considered is ', data[i][3])
                print('Considering the initial and final roll angles of', roll_angle_arr_val[0], 'and', roll_angle_arr_val[1], 'respectively, and yaw radius of', str(Ryaw[l]) + '.')
                print('The path type is ' + path_type + '. The minimum distance is', min_dist_path_length, '.')
                print('Runtime is ', runtime, 'seconds.')

                # We now save the data to a pickle file
                with open(data[i][3] + ' ' + 'ini_roll ' + str(roll_angle_arr_val[0]) + ' fin_roll ' + str(roll_angle_arr_val[1]) + ' Ryaw ' + str(Ryaw[l]) + '.pickle', 'wb') as f:
                    pickle.dump([min_dist_path_length, min_dist_path_pts, tang_global_path, tang_normal_global_path, surf_normal_global_path, path_type, runtime], f)

                # We now export the output to a csv file
                arr = [data[i][3], str(roll_angle_arr_val[0]), str(roll_angle_arr_val[1]), str(Ryaw[l]), str(min_dist_path_length), str(path_type), runtime]
                with open('results.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(arr)