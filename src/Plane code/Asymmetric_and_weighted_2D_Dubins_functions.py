# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 22:05:43 2022

@author: deepa
"""

import numpy as np
import math
import os
import copy
import matplotlib.pyplot as plt

# Importing the plotting class functions
plotting_class_path = 'D:\TAMU\Research\Cylinder code'
os.chdir(plotting_class_path)
from plotting_class import plotting_functions

# Changing the directory back to the original path
path_asym_functions = 'D:\TAMU\Research\Asymmetric_2D_Dubins\Codes'
os.chdir(path_asym_functions)

def ini_fin_config_manipulate(ini_config, fin_config):
    '''
    This function translates the initial configuration to the origin and updates
    the final configuration. The headings are also modified such that the final
    configuration is along the x-axis.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the coordinates of the initial position and the heading angle.
    fin_config : Numpy 1x3 array
        Contains the coordinates of the final position and the heading angle.

    Returns
    -------
    ini_config_mod : Numpy 1x3 array
        Contains the modified coordinates for the initial position such that it
        coincides with the origin, and the modified heading such that the final
        position is along the x-axis.
    fin_config_mod : Numpy 1x3 array
        Contains the modified coordinates for the initial position such that it
        coincides with the origin, and the modified heading such that the final
        position is along the x-axis.
    d : Scalar
        Eucledian distance between the initial and final positions
    
    '''
    
    # Angle of vector connecting initial and final configurations
    ang_vec = math.atan2((fin_config[1] - ini_config[1]),\
                         (fin_config[0] - ini_config[0]))
        
    # Length of the vector connecting the initial and final configurations
    d = math.sqrt((fin_config[0] - ini_config[0])**2 +\
                  (fin_config[1] - ini_config[1])**2)
    
    # Modified initial and final configurations
    ini_config_mod = np.array([0, 0, np.mod(ini_config[2] - ang_vec, 2*math.pi)])
    fin_config_mod = np.array([fin_config[0] - ini_config[0],\
                               fin_config[1] - ini_config[1],\
                               np.mod(fin_config[2] - ang_vec, 2*math.pi)])
        
    return ini_config_mod, fin_config_mod, d

def Seg_pts(start_pt_config, seg_param_val, rad_turn_seg, seg_type = 's'):
    
    '''
    This function returns an array of points corresponding to a left turn,
    right turn, or straight line segment, given the configuration corresponding
    to the start of the segment, the initial and final configurations in the original
    frame of reference, the arc length for the segment, the radius of the tight turn
    (used for left and right tight turns), and the type of the segment.
    The points returned are in the origin frame of reference.

    Parameters
    ----------
    start_pt_config : Numpy 1x3 array
        Configuration of the start of the arc in the original frame of reference.
    seg_param_val : Scalar
        Value of the parameter of the segment. If the segment is a straight line,
        then the parameter is the length of the straight line segment. If the segment
        is an arc of a circle, then the parameter of the segment is the angle of turn.
    rad_turn_seg : Scalar
        Radius of the tight turn which corresponds to the type of the turn. For
        example, the radius of the left and right turns need not be the same.
    seg_type : Character
        's' - straight line segment
        'l' - left tight turn, whose radius is given by rad_tight_turn
        'r' - right tight turn, whose radius if given by rad_tight_turn

    Returns
    -------
    pts_original_frame : Numpy nx3 array
        Contains the coordinates of the points along the left segment turn.
    
    '''
    
    # Discretizing the segment length for straight line or the angle for a tight turn
    segment_disc = np.linspace(0, seg_param_val, 100)
    pts_original_frame = np.zeros((100, 3))
    
    if seg_type.lower() == 'l' or seg_type.lower() == 'r':
    
        # Finding the coordinates of the point corresponding to the discretization
        # of the angle of the segment if left or right turn
        for i in range(len(segment_disc)):
            
            if seg_type.lower() == 'l':
        
                # x-coordinate
                pts_original_frame[i, 0] = start_pt_config[0]\
                    + rad_turn_seg*math.sin(start_pt_config[2] + segment_disc[i])\
                    - rad_turn_seg*math.sin(start_pt_config[2])
                # y-coordinate
                pts_original_frame[i, 1] = start_pt_config[1]\
                    - rad_turn_seg*math.cos(start_pt_config[2] + segment_disc[i])\
                    + rad_turn_seg*math.cos(start_pt_config[2])
                # Heading
                pts_original_frame[i, 2] = start_pt_config[2] + segment_disc[i]
                
            elif seg_type.lower() == 'r':
        
                # x-coordinate
                pts_original_frame[i, 0] = start_pt_config[0]\
                    - rad_turn_seg*math.sin(start_pt_config[2] - segment_disc[i])\
                    + rad_turn_seg*math.sin(start_pt_config[2])
                # y-coordinate
                pts_original_frame[i, 1] = start_pt_config[1]\
                    + rad_turn_seg*math.cos(start_pt_config[2] - segment_disc[i])\
                    - rad_turn_seg*math.cos(start_pt_config[2])
                # Heading
                pts_original_frame[i, 2] = start_pt_config[2] - segment_disc[i]
        
    elif seg_type.lower() == 's':
        
        # Finding the coordinates of the point corresponding to the discretization
        # on the line
        for i in range(len(segment_disc)):
            
            # x-coordinate
            pts_original_frame[i, 0] = start_pt_config[0] +\
                segment_disc[i]*math.cos(start_pt_config[2])
            # y-coordinate
            pts_original_frame[i, 1] = start_pt_config[1] +\
                segment_disc[i]*math.sin(start_pt_config[2])
            # Heading
            pts_original_frame[i, 2] = start_pt_config[2]
        
    return pts_original_frame

# def points_path(ini_config, fin_config, rL, rR, phi_1, param_seg_2, phi_3, path_type):
#     '''
#     This function returns points along the path in the original frame of reference
#     given the initial and final configurations, the radius of turn, the length of
#     each segment of the path, and the path type.

#     Parameters
#     ----------
#     ini_config : Numpy 1x3 array
#         Contains the initial configuration.
#     fin_config : Numpy 1x3 array
#         Contains the final configuration.
#     rL : Scalar
#         Radius of the left tight turn.
#     rR : Scalar
#         Radius of the right tight turn.
#     phi_1 : Scalar
#         Angle of the first segment of the path, which is a tight turn.
#     param_seg_2 : Scalar
#         Length (if straight line segment) or angle (if tight turn) of the second
#         segment of the path.
#     phi_3 : Scalar
#         Angle of the third segment of the path, which is a tight turn.
#     path_type : String
#         Contains the path type, and contains three segments made up of 'l' for left turn,
#         'r' for right turn, and 's' for straight line segment.

#     Returns
#     -------
#     x_coords_path : Numpy nx1 array
#         Contains the x-coordinate of points generated along the path.
#     y_coords_path : Numpy nx1 array
#         Contains the y-coordinate of points generated along the path.

#     '''
    
#     # Obtaining the points along the path
#     # Points along the first segment
#     if path_type[0].lower() == 'l':
    
#         pts_first_seg = Seg_pts(ini_config, phi_1, rL, path_type[0])
        
#     elif path_type[0].lower() == 'r':
    
#         pts_first_seg = Seg_pts(ini_config, phi_1, rR, path_type[0])
    
#     # Points along the second segment
#     if path_type[1].lower() == 'l':
    
#         pts_second_seg = Seg_pts(pts_first_seg[-1], param_seg_2, rL, path_type[1])
    
#     # The else condition also accounts for straight line segment, since for generating
#     # points along the straight line segment, the radius of the segment is not required.
#     else:
    
#         pts_second_seg = Seg_pts(pts_first_seg[-1], param_seg_2, rR, path_type[1])
        
#     # Points along the third segment
#     if path_type[2].lower() == 'l':
    
#         pts_third_seg = Seg_pts(pts_second_seg[-1], phi_3, rL, path_type[2])
    
#     elif path_type[2].lower() == 'r':
    
#         pts_third_seg = Seg_pts(pts_second_seg[-1], phi_3, rR, path_type[2])
        
#     # x coordinates for the complete path in the original frame of reference
#     x_coords_path = np.append(np.append(pts_first_seg[:, 0], pts_second_seg[:, 0]),\
#                               pts_third_seg[:, 0])
#     # y coordinates for the complete path in the original frame of reference
#     y_coords_path = np.append(np.append(pts_first_seg[:, 1], pts_second_seg[:, 1]),\
#                               pts_third_seg[:, 1])
#     # heading angles corresponding to the points along the path in the original
#     # frame of reference
#     # heading_path = np.append(np.append(pts_first_seg[:, 2], pts_second_seg[:, 2]),\
#     #                           pts_third_seg[:, 2])
    
#     return x_coords_path, y_coords_path

def points_path(ini_config, fin_config, rL, rR, params_seg, path_type):
    '''
    This function returns points along the path in the original frame of reference
    given the initial and final configurations, the radius of turn, the length of
    each segment of the path, and the path type.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    rL : Scalar
        Radius of the left tight turn.
    rR : Scalar
        Radius of the right tight turn.
    params_seg : Array of size equal to length of path_type string
        Contains the parameters of the segments of the path
    path_type : String
        Contains the path type, and contains segments made up of 'l' for left turn,
        'r' for right turn, and 's' for straight line segment.

    Returns
    -------
    x_coords_path : Numpy nx1 array
        Contains the x-coordinate of points generated along the path.
    y_coords_path : Numpy nx1 array
        Contains the y-coordinate of points generated along the path.

    '''
    
    # # Checking if the length of the path_type string is the same as the number
    # # of parameters provided in the params_seg array
    # if len(path_type) != len(params_seg):
        
    #     raise Exception('The number of parameters provided corresponding to the path ' + \
    #                     'do not match with the number of segments in the path')
    
    # Obtaining the points along the path
    config_after_ith_segment = ini_config
    
    x_coords_path = np.array([])
    y_coords_path = np.array([])
    for i in range(len(path_type)):
        
        # Obtaining the points along the ith segment
        if path_type[i].lower() == 'l':
        
            pts_ith_segment = Seg_pts(config_after_ith_segment, params_seg[i],\
                                      rL, 'l')
                
        elif path_type[i].lower() == 'r':
        
            pts_ith_segment = Seg_pts(config_after_ith_segment, params_seg[i],\
                                      rR, 'r')
                
        else:
            
            pts_ith_segment = Seg_pts(config_after_ith_segment, params_seg[i],\
                                      0, 's')
                
        # Appending the obtained points to the arrays
        x_coords_path = np.append(x_coords_path, pts_ith_segment[:, 0])
        y_coords_path = np.append(y_coords_path, pts_ith_segment[:, 1])
        
        # Updating the variable config_after_ith_segment
        config_after_ith_segment = pts_ith_segment[-1]
    
    return x_coords_path, y_coords_path

def CSC_path(ini_config, fin_config, rL, rR, muL, muR, path_type = 'lsl'):
    '''
    This function generates a CSC path connecting the initial and final configurations.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    rL : Scalar
        Radius of the left tight turn.
    rR : Scalar
        Radius of the right tight turn.
    muL : Scalar
        Penalty associated with a left turn.
    muR : Scalar
        Penalty associated with a right turn.
    path_type : String
        Contains the path type, and contains three segments made up of 'l' for left turn,
        'r' for right turn, and 's' for straight line segment.

    Returns
    -------
    path_length : Scalar
        Length of the path.
    cost_path : Scalar
        Cost of the path.
    phi_1 : Scalar
        Angle of the first segment of the path.
    lS : Scalar
        Length of the second segment of the path.
    phi_2 : Scalar
        Angle of the third segment of the path.

    '''
    
    # Initial and final configurations after modification, i.e., shifting the
    # initial configuration to the origin and the final configuration along
    # the x-axis
    ini_config_mod, fin_config_mod, d = ini_fin_config_manipulate(ini_config, fin_config)
    
    # Initial and final heading angles in the modified frame of reference
    alpha_i = ini_config_mod[2]
    alpha_f = fin_config_mod[2]
    
    path_types = ['lsl', 'rsr', 'lsr', 'rsl']
    
    if path_type.lower() not in path_types:
        
        raise Exception('Incorrect path type is provided')
    
    # Square of length for the straight line segment    
    if path_type.lower() == 'lsl':
    
        lS2 = d**2 + 2*rL**2 + 2*d*rL*(math.sin(alpha_i) - math.sin(alpha_f))\
            - 2*(rL**2)*math.cos(alpha_f - alpha_i)
        
    elif path_type.lower() == 'rsr':
        
        lS2 = d**2 + 2*rR**2 + 2*d*rR*(math.sin(alpha_f) - math.sin(alpha_i))\
            - 2*(rR**2)*math.cos(alpha_f - alpha_i)
            
    elif path_type.lower() == 'lsr':
        
        lS2 = d**2 + 2*d*(rL*math.sin(alpha_i) + rR*math.sin(alpha_f))\
            + 2*rL*rR*(math.cos(alpha_f - alpha_i) - 1)
            
    elif path_type.lower() == 'rsl':
        
        lS2 = d**2 - 2*d*(rR*math.sin(alpha_i) + rL*math.sin(alpha_f)) \
            + 2*rL*rR*(math.cos(alpha_f - alpha_i) - 1)
            
    # Accounting for numerical inaccuracies
    if lS2 < 0 and lS2 >= -10**(-6):
        
        lS2 = 0
            
    # Checking if the path exists
    if lS2 < 0:
        
        print(path_type.upper() + ' path does not exist.')
        path_length = np.NaN
        cost_path = np.NaN
        phi_1 = np.NaN
        lS = np.NaN
        phi_3 = np.NaN
        
    else:
        
        # Checking if the length of the straight line segment equals zero for
        # LSL and RSR paths. If yes, the paths become a degenerate L and R paths,
        # respectively.
        if lS2 <= 10**(-6) and path_type[0].lower() == path_type[2].lower():
            
            print('Path is of type ' + path_type[0].upper() + '.')
            
            if path_type[0].lower() == 'l':
                
                phi_1 = np.mod((alpha_f - alpha_i), 2*math.pi)
                lS = 0
                phi_3 = 0
                path_length = rL*phi_1
                cost_path = (rL + muL)*phi_1
                
            elif path_type[0].lower() == 'r':
                
                phi_1 = np.mod((alpha_i - alpha_f), 2*math.pi)
                lS = 0
                phi_3 = 0
                path_length = rR*phi_1
                cost_path = (rR + muR)*phi_1
                
        # Checking if the length of the straight line segment equals zero for
        # LSR and RSL paths. If yes, the paths become a degenerate LR or RL path
        elif lS2 <= 10**(-6):
            
            print('Path is of type '+ path_type[0].upper() + path_type[2].upper() + '.')
            
            if path_type[0].lower() == 'l':
                
                # Angle of the first arc
                # Here, delta := atan2((rL + rR), ls) is set as pi/2.
                phi_1 = np.mod(math.pi/2 - alpha_i\
                               + math.atan2(-(rL*math.cos(alpha_i) + rR*math.cos(alpha_f)),\
                                            d + rL*math.sin(alpha_i) + rR*math.sin(alpha_f)),\
                               2*math.pi)
                # Angle of the final arc
                phi_3 = np.mod((alpha_i - alpha_f + phi_1), 2*math.pi)
                # Length of the straight line segment
                lS = 0
                
                # Path length and cost
                path_length = rL*phi_1 + lS + rR*phi_3
                cost_path = (rL + muL)*phi_1 + lS + (rR + muR)*phi_3
                
            else:
                
                # Angle of the first arc
                # Here, delta := atan2((rL + rR), ls) is set as pi/2.
                phi_1 = np.mod(math.pi/2 + alpha_i\
                               - math.atan2(rR*math.cos(alpha_i) + rL*math.cos(alpha_f),\
                                            d - rR*math.sin(alpha_i) - rL*math.sin(alpha_f)),\
                               2*math.pi)
                # Angle of the final arc
                phi_3 = np.mod((alpha_f - alpha_i + phi_1), 2*math.pi)
                # Length of the straight line segment
                lS = 0
                
                # Path length and cost
                path_length = rR*phi_1 + lS + rL*phi_3
                cost_path = (rR + muR)*phi_1 + lS + (rL + muL)*phi_3
                
        else:
            
            # Length of the straight line segment
            lS = math.sqrt(lS2)
            
            # Computing the angles for the first and the last arcs
            # LSL path
            if path_type.lower() == 'lsl':
                
                # Angle of the first arc
                phi_1 = np.mod(math.atan2(rL*(math.cos(alpha_f) - math.cos(alpha_i)),\
                                          d - rL*(math.sin(alpha_f) - math.sin(alpha_i)))\
                               - alpha_i, 2*math.pi)
                # Angle of the final arc
                phi_3 = np.mod(alpha_f - alpha_i - phi_1, 2*math.pi)
                
                # Path length and cost
                path_length = rL*(phi_1 + phi_3) + lS
                cost_path = (rL + muL)*(phi_1 + phi_3) + lS
                
            # RSR path
            elif path_type.lower() == 'rsr':
                
                # Angle of the first arc
                phi_1 = np.mod(-math.atan2(-rR*(math.cos(alpha_f) - math.cos(alpha_i)),\
                                           d + rR*(math.sin(alpha_f) - math.sin(alpha_i)))\
                               + alpha_i, 2*math.pi)
                # Angle of the final arc
                phi_3 = np.mod(alpha_i - alpha_f - phi_1, 2*math.pi)
                
                # Path length and cost
                path_length = rR*(phi_1 + phi_3) + lS
                cost_path = (rR + muR)*(phi_1 + phi_3) + lS    
                
            # LSR path
            elif path_type.lower() == 'lsr':
                
                # Angle of the first arc
                phi_1 = np.mod(math.atan2((rL + rR), lS) - alpha_i\
                               + math.atan2(-(rL*math.cos(alpha_i) + rR*math.cos(alpha_f)),\
                                            d + rL*math.sin(alpha_i) + rR*math.sin(alpha_f)),\
                               2*math.pi)
                # Angle of the final arc
                phi_3 = np.mod((alpha_i - alpha_f + phi_1), 2*math.pi)
                
                # Path length and cost
                path_length = rL*phi_1 + lS + rR*phi_3
                cost_path = (rL + muL)*phi_1 + lS + (rR + muR)*phi_3
                
            # RSL path
            elif path_type.lower() == 'rsl':
                
                # Angle of the first arc
                phi_1 = np.mod(math.atan2((rL + rR), lS) + alpha_i\
                               - math.atan2(rR*math.cos(alpha_i) + rL*math.cos(alpha_f),\
                                            d - rR*math.sin(alpha_i) - rL*math.sin(alpha_f)),\
                               2*math.pi)
                # Angle of the final arc
                phi_3 = np.mod((alpha_f - alpha_i + phi_1), 2*math.pi)
                
                # Path length and cost
                path_length = rR*phi_1 + lS + rL*phi_3
                cost_path = (rR + muR)*phi_1 + lS + (rL + muL)*phi_3
        
    return path_length, cost_path, phi_1, lS, phi_3

# def CCC_path(ini_config, fin_config, rL, rR, muL, muR, path_type = 'lrl'):
#     '''
#     This function generates a CCC path connecting the initial and final configurations.
#     Note that if a CCC path exists, there are two paths of the same type wherein in one
#     path, the angle of the middle arc is less than pi, and in the other, the angle of the
#     middle arc is greater than pi. This function always returns parameters of the path that
#     has the middle arc angle greater than pi since from PMP, it was proved that the middle arc
#     angle is greater than pi in the optimal path for the unweighted Dubins problem.

#     Parameters
#     ----------
#     ini_config : Numpy 1x3 array
#         Contains the initial configuration.
#     fin_config : Numpy 1x3 array
#         Contains the final configuration.
#     rL : Scalar
#         Radius of the left tight turn.
#     rR : Scalar
#         Radius of the right tight turn.
#     muL : Scalar
#         Penalty associated with a left turn.
#     muR : Scalar
#         Penalty associated with a right turn.
#     path_type : String
#         Contains the path type, and contains three segments made up of 'l' for left turn,
#         'r' for right turn, and 's' for straight line segment.

#     Returns
#     -------
#     path_length : Scalar
#         Length of the path.
#     cost_path : Scalar
#         Cost of the path.
#     phi_1 : Scalar
#         Angle of the first segment of the path.
#     phi_2 : Scalar
#         Angle of the second segment of the path.
#     phi_3 : Scalar
#         Angle of the third segment of the path.
        
#     '''
#     # Initial and final configurations after modification, i.e., shifting the
#     # initial configuration to the origin and the final configuration along
#     # the x-axis
#     ini_config_mod, fin_config_mod, d = ini_fin_config_manipulate(ini_config, fin_config)
    
#     # Initial and final heading angles in the modified frame of reference
#     alpha_i = ini_config_mod[2]
#     alpha_f = fin_config_mod[2]
    
#     if path_type.lower() != 'lrl' and path_type.lower() != 'rlr':
        
#         raise Exception('Incorrect path type is provided')
    
#     # Angle calculation corresponding to the middle arc
#     if path_type.lower() == 'lrl':
        
#         cos_phi_2 = 1 - (1/(2*(rL + rR)**2))*(d**2 + 2*rL**2\
#                                               - 2*d*rL*(math.sin(alpha_f) - math.sin(alpha_i))\
#                                               - 2*(rL**2)*math.cos(alpha_f - alpha_i))
            
#     elif path_type.lower() == 'rlr':
        
#         cos_phi_2 = 1 - (1/(2*(rL + rR)**2))*(d**2 + 2*rR**2\
#                                               + 2*d*rR*(math.sin(alpha_f) - math.sin(alpha_i))\
#                                               - 2*(rR**2)*math.cos(alpha_f - alpha_i))
            
#     # Checking if the path exists
#     if abs(cos_phi_2) > 1:
        
#         print(path_type.upper() + ' path does not exist.')
#         path_length = np.NaN
#         cost_path = np.NaN
#         phi_1 = np.NaN
#         phi_2 = np.NaN
#         phi_3 = np.NaN
        
#     # Checking if the path is a degenerate 'C' path
#     elif cos_phi_2 == 1:
        
#         print('Path is of type ' + path_type[0].upper())
        
#         # Setting the second and third angles to zero. Third angle is set to zero
#         # since the first and third arcs of the same type.
#         phi_2 = 0
#         phi_3 = 0
        
#         if path_type.lower() == 'lrl':
            
#             phi_1 = np.mod((alpha_f - alpha_i), 2*math.pi)
#             path_length = rL*phi_1
#             cost_path = (rL + muL)*phi_1
            
#         elif path_type.lower() == 'rlr':
            
#             phi_1 = np.mod((alpha_i - alpha_f), 2*math.pi)
#             path_length = rR*phi_1
#             cost_path = (rR + muR)*phi_1
            
#     else:
        
#         # Choosing phi_2 such that it is always greater than pi. In the case that
#         # cos_phi_2 = -1, phi_2 = pi.
#         phi_2 = 2*math.pi - math.acos(cos_phi_2)
        
#         # Computing the angles of the first and last segments
#         # LRL path
#         if path_type.lower() == 'lrl':
            
#             # Angle of the first arc
#             phi_1 = np.mod(math.atan2(rL*(math.cos(alpha_f) - math.cos(alpha_i)),\
#                                       d - rL*(math.sin(alpha_f) - math.sin(alpha_i)))\
#                            - alpha_i + phi_2/2, 2*math.pi)
#             # Angle of the final arc
#             phi_3 = np.mod((alpha_f - alpha_i - phi_1 + phi_2), 2*math.pi)
            
#             # Path length and cost
#             path_length = rL*(phi_1 + phi_3) + rR*phi_2
#             cost_path = (rL + muL)*(phi_1 + phi_3) + (rR + muR)*phi_2
            
#         elif path_type.lower() == 'rlr':
            
#             # Angle of the first arc
#             phi_1 = np.mod(-math.atan2(-rR*(math.cos(alpha_f) - math.cos(alpha_i)),\
#                                        d + rR*(math.sin(alpha_f) - math.sin(alpha_i)))\
#                            + alpha_i + phi_2/2, 2*math.pi)            
#             # Angle of the final arc
#             phi_3 = np.mod((alpha_i - alpha_f - phi_1 + phi_2), 2*math.pi)
            
#             # Path length and cost
#             path_length = rR*(phi_1 + phi_3) + rL*phi_2
#             cost_path = (rR + muR)*(phi_1 + phi_3) + (rL + muL)*phi_2
        
#     return path_length, cost_path, phi_1, phi_2, phi_3

def CCC_path(ini_config, fin_config, rL, rR, muL, muR, path_type = 'lrl'):
    '''
    This function generates a CCC path connecting the initial and final configurations.
    Note that if a CCC path exists, there are two paths of the same type wherein in one
    path, the angle of the middle arc is less than pi, and in the other, the angle of the
    middle arc is greater than pi. This function always returns parameters of the path that
    has the middle arc angle greater than pi since from PMP, it was proved that the middle arc
    angle is greater than pi in the optimal path for the unweighted Dubins problem.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    rL : Scalar
        Radius of the left tight turn.
    rR : Scalar
        Radius of the right tight turn.
    muL : Scalar
        Penalty associated with a left turn.
    muR : Scalar
        Penalty associated with a right turn.
    path_type : String
        Contains the path type, and contains three segments made up of 'l' for left turn,
        'r' for right turn, and 's' for straight line segment.

    Returns
    -------
    path_length : Scalar
        Length of the path.
    cost_path : Scalar
        Cost of the path.
    phi_1 : Scalar
        Angle of the first segment of the path.
    phi_2 : Scalar
        Angle of the second segment of the path.
    phi_3 : Scalar
        Angle of the third segment of the path.
        
    '''
    # Initial and final configurations after modification, i.e., shifting the
    # initial configuration to the origin and the final configuration along
    # the x-axis
    ini_config_mod, fin_config_mod, d = ini_fin_config_manipulate(ini_config, fin_config)
    
    # Initial and final heading angles in the modified frame of reference
    alpha_i = ini_config_mod[2]
    alpha_f = fin_config_mod[2]
    
    if path_type.lower() != 'lrl' and path_type.lower() != 'rlr':
        
        raise Exception('Incorrect path type is provided')
    
    # Angle calculation corresponding to the middle arc
    if path_type.lower() == 'lrl':
        
        cos_phi_2 = 1 - (1/(2*(rL + rR)**2))*(d**2 + 2*rL**2\
                                              - 2*d*rL*(math.sin(alpha_f) - math.sin(alpha_i))\
                                              - 2*(rL**2)*math.cos(alpha_f - alpha_i))
            
    elif path_type.lower() == 'rlr':
        
        cos_phi_2 = 1 - (1/(2*(rL + rR)**2))*(d**2 + 2*rR**2\
                                              + 2*d*rR*(math.sin(alpha_f) - math.sin(alpha_i))\
                                              - 2*(rR**2)*math.cos(alpha_f - alpha_i))
            
    # Accounting for numerical inaccuracies
    if abs(cos_phi_2) > 1 and abs(cos_phi_2) <= 1 + 10**(-6):
        
        cos_phi_2 = np.sign(cos_phi_2)
            
    # Checking if the path exists
    if abs(cos_phi_2) > 1:
        
        print(path_type.upper() + ' path does not exist.')
        path_length = np.NaN
        cost_path = np.NaN
        phi_1 = np.NaN
        phi_2 = np.NaN
        phi_3 = np.NaN
        
    # Checking if the path is a degenerate 'C' path
    elif cos_phi_2 == 1:
        
        print('Path is of type ' + path_type[0].upper())
        
        # Setting the second and third angles to zero. Third angle is set to zero
        # since the first and third arcs of the same type.
        # Note that since a degenerate 'C' path is accounted for in the CSC path
        # type, it is not generated for a CCC path
        phi_2 = np.NaN
        phi_3 = np.NaN
        phi_1 = np.NaN
        path_length = np.NaN
        cost_path = np.NaN
            
    else:
        
        # Choosing phi_2 such that it is always greater than pi. In the case that
        # cos_phi_2 = -1, phi_2 = pi.
        phi_2 = 2*math.pi - math.acos(cos_phi_2)
        
        # Computing the angles of the first and last segments
        # LRL path
        if path_type.lower() == 'lrl':
            
            # Angle of the first arc
            phi_1 = np.mod(math.atan2(rL*(math.cos(alpha_f) - math.cos(alpha_i)),\
                                      d - rL*(math.sin(alpha_f) - math.sin(alpha_i)))\
                           - alpha_i + phi_2/2, 2*math.pi)
            # Angle of the final arc
            phi_3 = np.mod((alpha_f - alpha_i - phi_1 + phi_2), 2*math.pi)
            
            # Path length and cost
            path_length = rL*(phi_1 + phi_3) + rR*phi_2
            cost_path = (rL + muL)*(phi_1 + phi_3) + (rR + muR)*phi_2
            
        elif path_type.lower() == 'rlr':
            
            # Angle of the first arc
            phi_1 = np.mod(-math.atan2(-rR*(math.cos(alpha_f) - math.cos(alpha_i)),\
                                       d + rR*(math.sin(alpha_f) - math.sin(alpha_i)))\
                           + alpha_i + phi_2/2, 2*math.pi)            
            # Angle of the final arc
            phi_3 = np.mod((alpha_i - alpha_f - phi_1 + phi_2), 2*math.pi)
            
            # Path length and cost
            path_length = rR*(phi_1 + phi_3) + rL*phi_2
            cost_path = (rR + muR)*(phi_1 + phi_3) + (rL + muL)*phi_2
            
        # Removing degenerate CCC paths
        if phi_1 <= 10**(-6) or phi_2 <= 10**(-6) or phi_3 <= 10**(-6):
            
            phi_1 = np.NaN
            phi_2 = np.NaN
            phi_3 = np.NaN
            path_length = np.NaN
            cost_path = np.NaN
        
    return path_length, cost_path, phi_1, phi_2, phi_3

def SCS_path(ini_config, fin_config, rturn, mu, path_type = 'sls'):
    '''
    This function generates a SCS path connecting the initial and final configurations.
    An SCS path is generated only if the angle of the middle turn is greater than
    pi.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    rturn : Scalar
        Radius of the tight turn.
    mu : Scalar
        Penalty associated with the tight turn.
    path_type : String
        Contains the SCS path type, where the middle turn is a left or a right turn.

    Returns
    -------
    path_length : Scalar
        Length of the path.
    cost_path : Scalar
        Cost of the path.
    s1 : Scalar
        Length of the first straight line segment of the path.
    phi_2 : Scalar
        Angle of the second segment of the path.
    s3 : Scalar
        Length of the second straight line segment of the path.
        
    '''
    
    # Initial and final configurations after modification, i.e., shifting the
    # initial configuration to the origin and the final configuration along
    # the x-axis
    ini_config_mod, fin_config_mod, d = ini_fin_config_manipulate(ini_config, fin_config)
    
    # Initial and final heading angles in the modified frame of reference
    alpha_i = ini_config_mod[2]
    alpha_f = fin_config_mod[2]
    
    path_types = ['sls', 'srs']
    
    if path_type.lower() not in path_types:
        
        raise Exception('Incorrect path type is provided')
    
    # Checking the existence of an SCS path
    if abs(math.sin(alpha_f - alpha_i)) <= 10**(-6):
        
        print('An ' + str(path_type.upper()) + ' path does not exist')
        s1 = np.NaN
        phi_2 = np.NAN
        s3 = np.NaN
        path_length = np.NaN
        cost_path = np.NaN
        
    elif (path_type[1].lower() == 'l' and np.mod(alpha_f - alpha_i, 2*math.pi) <= math.pi)\
        or (path_type[1].lower() == 'r' and np.mod(alpha_i - alpha_f, 2*math.pi) <= math.pi):
        
        print('An optimal ' + str(path_type.upper()) + ' path does not exist')
        s1 = np.NaN
        phi_2 = np.NAN
        s3 = np.NaN
        path_length = np.NaN
        cost_path = np.NaN
        
    else:
        
        # Obtaining the parameters of the path
        if path_type[1].lower() == 'l':
        
            phi_2 = np.mod(alpha_f - alpha_i, 2*math.pi)
            s1 = (1/math.sin(alpha_f - alpha_i))*(d*math.sin(alpha_f)\
                                                  - rturn*(1 - math.cos(alpha_f - alpha_i)))
            s3 = (1/math.sin(alpha_f - alpha_i))*(-d*math.sin(alpha_i)\
                                                  - rturn*(1 - math.cos(alpha_f - alpha_i)))
                
        elif path_type[1].lower() == 'r':
        
            phi_2 = np.mod(alpha_i - alpha_f, 2*math.pi)
            s1 = (1/math.sin(alpha_f - alpha_i))*(d*math.sin(alpha_f)\
                                                  + rturn*(1 - math.cos(alpha_f - alpha_i)))
            s3 = (1/math.sin(alpha_f - alpha_i))*(-d*math.sin(alpha_i)\
                                                  + rturn*(1 - math.cos(alpha_f - alpha_i)))
                
        # Computing the path length and the path cost
        path_length = s1 + rturn*phi_2 + s3
        cost_path = s1 + (rturn + mu)*phi_2 + s3
        
        # Checking the existence of a non-degenerate SCS path, or if the length
        # of a straight line segment is negative
        if s1 <= 10**(-6) or phi_2 <= 10**(-6) or s3 <= 10**(-6):
            
            print('The optimal ' + str(path_type.upper()) + ' path is degenerate or does not exist')
            s1 = np.NaN
            phi_2 = np.NaN
            s3 = np.NaN
            path_length = np.NaN
            cost_path = np.NaN
        
    return path_length, cost_path, s1, phi_2, s3

# def SCSC_CSCS_paths(ini_config, fin_config, rL, rR, muL, muR, path_type = 'lsrs'):
#     '''
#     This function generates a CSCS or SCSC path connecting the initial and final
#     configurations.

#     Parameters
#     ----------
#     ini_config : Numpy 1x3 array
#         Contains the initial configuration.
#     fin_config : Numpy 1x3 array
#         Contains the final configuration.
#     rL : Scalar
#         Radius of the left tight turn.
#     rR : Scalar
#         Radius of the right tight turn.
#     muL : Scalar
#         Penalty associated with a left turn.
#     muR : Scalar
#         Penalty associated with a right turn.
#     path_type : String
#         Contains the path type, and contains five segments made up of 'l' for left turn,
#         'r' for right turn, and 's' for straight line segment. ALlowed path types are
#         LSRS, SRSL, RSLS, and SLSR.

#     Returns
#     -------
#     path_length : Scalar
#         Length of the path.
#     cost_path : Scalar
#         Cost of the path.
#     p1 : Scalar
#         Parameter corresponding to the first segment of the path. If the first
#         segment is S, then the parameter is the length of the first segment.
#         If the segment is C, then the parameter is the angle of the first segment.
#         The same syntax is followed for p2, p3, and p4.
#     p2 : Scalar
#         Parameter corresponding to the second segment of the path.
#     p3 : Scalar
#         Parameter corresponding to the third segment of the path.
#     p4 : Scalar
#         Parameter corresponding to the fourth segment of the path.
#     '''
    
#     # Initial and final configurations after modification, i.e., shifting the
#     # initial configuration to the origin and the final configuration along
#     # the x-axis
#     ini_config_mod, fin_config_mod, d = ini_fin_config_manipulate(ini_config, fin_config)
    
#     # Initial and final heading angles in the modified frame of reference
#     alpha_i = ini_config_mod[2]
#     alpha_f = fin_config_mod[2]
    
#     path_types = ['lsrs', 'srsl', 'rsls', 'slsr']
    
#     if path_type.lower() not in path_types:
        
#         raise Exception('Incorrect path type is provided')
    
#     # Initializing the return variables as np.NaN. If a non-trivial LSRSL path exists,
#     # these variables are overwritten.
#     path_length = np.NaN
#     cost_path = np.NaN
#     p1 = np.NaN
#     p2 = np.NaN
#     p3 = np.NaN
#     p4 = np.NaN
    
#     # Creating arrays to store parameters corresponding to two possible solutions
#     p1_solns_array = []
#     p2_solns_array = []
#     p3_solns_array = []
#     p4_solns_array = []
#     path_length_array = []
#     path_cost_array = []
    
#     # Checking for existence of singular solution and computing the corresponding
#     # solution if it exists
#     if path_type.lower() == 'lsrs':
        
#         if abs(alpha_f - np.mod(math.atan2((-rL*math.cos(alpha_i) - rR*math.cos(alpha_f)),\
#                                            (d + rL*math.sin(alpha_i) + rR*math.sin(alpha_f))),\
#                                 2*math.pi)) <= 10**(-6):
            
#             # Computing the parameters corresponding to the singular solution,
#             # and allocating it to temporary variables
#             p3t = math.pi + 2*math.asin(math.sqrt((rL + rR)/(2*(rL + rR + muL + muR))))
#             p2t = -(muL + muR)*(1/math.tan(p3t/2))
#             p1t = np.mod((alpha_f - alpha_i + p3t), 2*math.pi)
            
#             if abs(math.sin(alpha_f)) <= 10**(-2):
                
#                 # Using the first of the two equations to compute p4
#                 p4t = (1/math.cos(alpha_f))*(d + rL*math.sin(alpha_i) + rR*math.sin(alpha_f))\
#                     + math.sqrt(p2t**2 + (rL + rR)**2)
                    
#             else:
                
#                 # Using the second of the two equations to compute p4
#                 p4t = (1/math.sin(alpha_f))*(-rL*math.cos(alpha_i) - rR*math.cos(alpha_f))\
#                     + math.sqrt(p2t**2 + (rL + rR)**2)
                    
#             # Checking if the singular solution is a legitimate solution
#             if p4t >= 10**(-6) and p1t >= 10**(-6):
                
#                 p1_solns_array.append(p1t)
#                 p2_solns_array.append(p2t)
#                 p3_solns_array.append(p3t)
#                 p4_solns_array.append(p4t)
#                 path_length_array.append((rL*p1t + p2t + rR*p3t + p4t))
#                 path_cost_array.append(((rL + muL)*p1t + p2t + (rR + muR)*p3t + p4t))
                
#     elif path_type.lower() == 'rsls':
        
#         if abs(alpha_f - np.mod(math.atan2((rR*math.cos(alpha_i) + rL*math.cos(alpha_f)),\
#                                            (d - rR*math.sin(alpha_i) - rL*math.sin(alpha_f))),\
#                                 2*math.pi)) <= 10**(-6):
            
#             # Computing the parameters corresponding to the singular solution,
#             # and allocating it to temporary variables
#             p3t = math.pi + 2*math.asin(math.sqrt((rL + rR)/(2*(rL + rR + muL + muR))))
#             p2t = -(muL + muR)*(1/math.tan(p3t/2))
#             p1t = np.mod((alpha_i - alpha_f + p3t), 2*math.pi)
            
#             if abs(math.sin(alpha_f)) <= 10**(-2):
                
#                 # Using the first of the two equations to compute p4
#                 p4t = (1/math.cos(alpha_f))*(d - rR*math.sin(alpha_i) - rL*math.sin(alpha_f))\
#                     + math.sqrt(p2t**2 + (rL + rR)**2)
                    
#             else:
                
#                 # Using the second of the two equations to compute p4
#                 p4t = (1/math.sin(alpha_f))*(rR*math.cos(alpha_i) + rL*math.cos(alpha_f))\
#                     + math.sqrt(p2t**2 + (rL + rR)**2)
                    
#             # Checking if the singular solution is a legitimate solution
#             if p4t >= 10**(-6) and p1t >= 10**(-6):
                
#                 p1_solns_array.append(p1t)
#                 p2_solns_array.append(p2t)
#                 p3_solns_array.append(p3t)
#                 p4_solns_array.append(p4t)
#                 path_length_array.append((rR*p1t + p2t + rL*p3t + p4t))
#                 path_cost_array.append(((rR + muR)*p1t + p2t + (rL + muL)*p3t + p4t))
                
#     elif path_type.lower() == 'slsr':
        
#         if abs(alpha_i - np.mod(math.atan2((-rL*math.cos(alpha_i) - rR*math.cos(alpha_f)),\
#                                            (d + rL*math.sin(alpha_i) + rR*math.sin(alpha_f))),\
#                                 2*math.pi)) <= 10**(-6):
            
#             # Computing the parameters corresponding to the singular solution,
#             # and allocating it to temporary variables
#             p2t = math.pi + 2*math.asin(math.sqrt((rL + rR)/(2*(rL + rR + muL + muR))))
#             p3t = -(muL + muR)*(1/math.tan(p2t/2))
#             p4t = np.mod((alpha_i - alpha_f + p2t), 2*math.pi)
            
#             if abs(math.sin(alpha_i)) <= 10**(-2):
                
#                 # Using the first of the two equations to compute p1
#                 p1t = (1/math.cos(alpha_i))*(d + rL*math.sin(alpha_i) + rR*math.sin(alpha_f))\
#                     + math.sqrt(p3t**2 + (rL + rR)**2)
                    
#             else:
                
#                 # Using the second of the two equations to compute p1
#                 p1t = (1/math.sin(alpha_i))*(-rL*math.cos(alpha_i) - rR*math.cos(alpha_f))\
#                     + math.sqrt(p3t**2 + (rL + rR)**2)
                    
#             # Checking if the singular solution is a legitimate solution
#             if p1t >= 10**(-6) and p4t >= 10**(-6):
                
#                 p1_solns_array.append(p1t)
#                 p2_solns_array.append(p2t)
#                 p3_solns_array.append(p3t)
#                 p4_solns_array.append(p4t)
#                 path_length_array.append((p1t + rL*p2t + p3t + rR*p4t))
#                 path_cost_array.append((p1t + (rL + muL)*p2t + p3t + (rR + muR)*p4t))
                
#     elif path_type.lower() == 'srsl':
        
#         if abs(alpha_i - np.mod(math.atan2((rR*math.cos(alpha_i) + rL*math.cos(alpha_f)),\
#                                            (d - rR*math.sin(alpha_i) - rL*math.sin(alpha_f))),\
#                                 2*math.pi)) <= 10**(-6):
            
#             # Computing the parameters corresponding to the singular solution,
#             # and allocating it to temporary variables
#             p2t = math.pi + 2*math.asin(math.sqrt((rL + rR)/(2*(rL + rR + muL + muR))))
#             p3t = -(muL + muR)*(1/math.tan(p2t/2))
#             p4t = np.mod((alpha_f - alpha_i + p2t), 2*math.pi)
            
#             if abs(math.sin(alpha_i)) <= 10**(-2):
                
#                 # Using the first of the two equations to compute p1
#                 p1t = (1/math.cos(alpha_i))*(d - rR*math.sin(alpha_i) - rL*math.sin(alpha_f))\
#                     + math.sqrt(p3t**2 + (rL + rR)**2)
                    
#             else:
                
#                 # Using the second of the two equations to compute p1
#                 p1t = (1/math.sin(alpha_i))*(rR*math.cos(alpha_i) + rL*math.cos(alpha_f))\
#                     + math.sqrt(p3t**2 + (rL + rR)**2)
                    
#             # Checking if the singular solution is a legitimate solution
#             if p1t >= 10**(-6) and p4t >= 10**(-6):
                
#                 p1_solns_array.append(p1t)
#                 p2_solns_array.append(p2t)
#                 p3_solns_array.append(p3t)
#                 p4_solns_array.append(p4t)
#                 path_length_array.append((p1t + rR*p2t + p3t + rL*p4t))
#                 path_cost_array.append((p1t + (rR + muR)*p2t + p3t + (rL + muL)*p4t))
                
#     # Computing the non-singular solution
#     if path_type.lower() == 'lsrs':
        
#         c = d*math.sin(alpha_f) + rL*math.cos(alpha_f - alpha_i) + rR
#         sin2p3RHS = (rL + rR + 2*(muL + muR) - c)/(2*(rL + rR + muL + muR))
        
#         if sin2p3RHS > 0 and sin2p3RHS < 1:
            
#             p3t = 2*math.pi - 2*math.asin(math.sqrt(sin2p3RHS))
#             p2t = -(muL + muR)*(1/math.tan(p3t/2))
#             p1t = np.mod((alpha_f - alpha_i + p3t), 2*math.pi)
#             delta = math.atan2((rL + rR), p2t)
#             # Computing the solution corresponding to p4
#             if abs(math.sin(-p3t + delta)) >= 10**(-6):
                
#                 p4t = (1/(math.sin(-p3t + delta)))*(-d*math.sin(alpha_f + p3t - delta)\
#                                                     -rL*math.cos(p1t - delta)\
#                                                     - rR*math.cos(p3t - delta))
                    
#                 # Checking if the solution set is valid
#                 if p4t >= 10**(-6) and p1t >= 10**(-6):
                    
#                     p1_solns_array.append(p1t)
#                     p2_solns_array.append(p2t)
#                     p3_solns_array.append(p3t)
#                     p4_solns_array.append(p4t)
#                     path_length_array.append((rL*p1t + p2t + rR*p3t + p4t))
#                     path_cost_array.append(((rL + muL)*p1t + p2t + (rR + muR)*p3t + p4t))
                    
#     elif path_type.lower() == 'rsls':
        
#         c = -d*math.sin(alpha_f) + rR*math.cos(alpha_f - alpha_i) + rL
#         sin2p3RHS = (rL + rR + 2*(muL + muR) - c)/(2*(rL + rR + muL + muR))
        
#         if sin2p3RHS > 0 and sin2p3RHS < 1:
            
#             p3t = 2*math.pi - 2*math.asin(math.sqrt(sin2p3RHS))
#             p2t = -(muL + muR)*(1/math.tan(p3t/2))
#             p1t = np.mod((alpha_i - alpha_f + p3t), 2*math.pi)
#             delta = math.atan2((rL + rR), p2t)
#             # Computing the solution corresponding to p4
#             if abs(math.sin(p3t - delta)) >= 10**(-6):
                
#                 p4t = (1/(math.sin(p3t - delta)))*(-d*math.sin(alpha_f - p3t + delta)\
#                                                    + rR*math.cos(-p1t + delta)\
#                                                    + rL*math.cos(-p3t + delta))
                    
#                 # Checking if the solution set is valid
#                 if p4t >= 10**(-6) and p1t >= 10**(-6):
                    
#                     p1_solns_array.append(p1t)
#                     p2_solns_array.append(p2t)
#                     p3_solns_array.append(p3t)
#                     p4_solns_array.append(p4t)
#                     path_length_array.append((rR*p1t + p2t + rL*p3t + p4t))
#                     path_cost_array.append(((rR + muR)*p1t + p2t + (rL + muL)*p3t + p4t))
                    
#     elif path_type.lower() == 'slsr':
        
#         c = d*math.sin(alpha_i) + rR*math.cos(alpha_f - alpha_i) + rL
#         sin2p2RHS = (rL + rR + 2*(muL + muR) - c)/(2*(rL + rR + muL + muR))
        
#         if sin2p2RHS > 0 and sin2p2RHS < 1:
            
#             p2t = 2*math.pi - 2*math.asin(math.sqrt(sin2p2RHS))
#             p3t = -(muL + muR)*(1/math.tan(p2t/2))
#             p4t = np.mod((alpha_i - alpha_f + p2t), 2*math.pi)
#             delta = math.atan2((rL + rR), p3t)
#             # Computing the solution corresponding to p1
#             if abs(math.sin(-p2t + delta)) >= 10**(-6):
                
#                 p1t = (1/(math.sin(-p2t + delta)))*(-d*math.sin(alpha_i + p2t - delta)\
#                                                     -rR*math.cos(p4t - delta)\
#                                                     - rL*math.cos(p2t - delta))
                    
#                 # Checking if the solution set is valid
#                 if p4t >= 10**(-6) and p1t >= 10**(-6):
                    
#                     p1_solns_array.append(p1t)
#                     p2_solns_array.append(p2t)
#                     p3_solns_array.append(p3t)
#                     p4_solns_array.append(p4t)
#                     path_length_array.append((p1t + rL*p2t + p3t + rR*p4t))
#                     path_cost_array.append((p1t + (rL + muL)*p2t + p3t + (rR + muR)*p4t))
                    
#     elif path_type.lower() == 'srsl':
        
#         c = -d*math.sin(alpha_i) + rL*math.cos(alpha_f - alpha_i) + rR
#         sin2p2RHS = (rL + rR + 2*(muL + muR) - c)/(2*(rL + rR + muL + muR))
        
#         if sin2p2RHS > 0 and sin2p2RHS < 1:
            
#             p2t = 2*math.pi - 2*math.asin(math.sqrt(sin2p2RHS))
#             p3t = -(muL + muR)*(1/math.tan(p2t/2))
#             p4t = np.mod((alpha_f - alpha_i + p2t), 2*math.pi)
#             delta = math.atan2((rL + rR), p3t)
#             # Computing the solution corresponding to p1
#             if abs(math.sin(p2t - delta)) >= 10**(-6):
                
#                 p1t = (1/(math.sin(p2t - delta)))*(-d*math.sin(alpha_i - p2t + delta)\
#                                                    + rL*math.cos(-p4t + delta)\
#                                                    + rR*math.cos(-p2t + delta))
                    
#                 # Checking if the solution set is valid
#                 if p4t >= 10**(-6) and p1t >= 10**(-6):
                    
#                     p1_solns_array.append(p1t)
#                     p2_solns_array.append(p2t)
#                     p3_solns_array.append(p3t)
#                     p4_solns_array.append(p4t)
#                     path_length_array.append((p1t + rR*p2t + p3t + rL*p4t))
#                     path_cost_array.append((p1t + (rR + muR)*p2t + p3t + (rL + muL)*p4t))
                    
#     # Computing the minimum cost solution among the two possible solutions
#     if len(path_cost_array) != 0: # If at least one solution exists
            
#             # Obtaining the index corresponding to the minimum cost solution
#             ind_min_cost_soln = np.nanargmin(path_cost_array)
#             path_length = path_length_array[ind_min_cost_soln]
#             cost_path = path_cost_array[ind_min_cost_soln]
#             p1 = p1_solns_array[ind_min_cost_soln]
#             p2 = p2_solns_array[ind_min_cost_soln]
#             p3 = p3_solns_array[ind_min_cost_soln]
#             p4 = p4_solns_array[ind_min_cost_soln]
    
#     return path_length, cost_path, p1, p2, p3, p4

def SCSC_CSCS_paths(ini_config, fin_config, rL, rR, muL, muR, path_type = 'lsrs'):
    '''
    This function generates a CSCS or SCSC path connecting the initial and final
    configurations.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    rL : Scalar
        Radius of the left tight turn.
    rR : Scalar
        Radius of the right tight turn.
    muL : Scalar
        Penalty associated with a left turn.
    muR : Scalar
        Penalty associated with a right turn.
    path_type : String
        Contains the path type, and contains five segments made up of 'l' for left turn,
        'r' for right turn, and 's' for straight line segment. ALlowed path types are
        LSRS, SRSL, RSLS, and SLSR.

    Returns
    -------
    path_length : Scalar
        Length of the path.
    cost_path : Scalar
        Cost of the path.
    p1 : Scalar
        Parameter corresponding to the first segment of the path. If the first
        segment is S, then the parameter is the length of the first segment.
        If the segment is C, then the parameter is the angle of the first segment.
        The same interpretation is used for p2, p3, and p4.
    p2 : Scalar
        Parameter corresponding to the second segment of the path.
    p3 : Scalar
        Parameter corresponding to the third segment of the path.
    p4 : Scalar
        Parameter corresponding to the fourth segment of the path.
    '''
    
    # Initial and final configurations after modification, i.e., shifting the
    # initial configuration to the origin and the final configuration along
    # the x-axis
    ini_config_mod, fin_config_mod, d = ini_fin_config_manipulate(ini_config, fin_config)
    
    # Initial and final heading angles in the modified frame of reference
    alpha_i = ini_config_mod[2]
    alpha_f = fin_config_mod[2]
    
    path_types = ['lsrs', 'srsl', 'rsls', 'slsr']
    
    if path_type.lower() not in path_types:
        
        raise Exception('Incorrect path type is provided')
    
    # Initializing the return variables as np.NaN. If a non-trivial LSRSL path exists,
    # these variables are overwritten.
    path_length = np.NaN
    cost_path = np.NaN
    p1 = np.NaN
    p2 = np.NaN
    p3 = np.NaN
    p4 = np.NaN
    
    # Creating arrays to store parameters corresponding to two possible solutions
    p1_solns_array = []
    p2_solns_array = []
    p3_solns_array = []
    p4_solns_array = []
    path_length_array = []
    path_cost_array = []
    
    # Checking for existence of singular solution and computing the corresponding
    # solution if it exists
    if path_type.lower() == 'lsrs':
            
        # Computing the parameters corresponding to the singular solution,
        # and allocating it to temporary variables
        p3t = math.pi + 2*math.asin(math.sqrt((rL + rR)/(2*(rL + rR + muL + muR))))
        p2t = -(muL + muR)*(1/math.tan(p3t/2))
        p1t = np.mod((alpha_f - alpha_i + p3t), 2*math.pi)
        
        if abs(math.sin(alpha_f)) <= 10**(-2):
            
            # Using the first of the two equations to compute p4
            p4t = (1/math.cos(alpha_f))*(d + rL*math.sin(alpha_i) + rR*math.sin(alpha_f))\
                + math.sqrt(p2t**2 + (rL + rR)**2)
                
        else:
            
            # Using the second of the two equations to compute p4
            p4t = (1/math.sin(alpha_f))*(-rL*math.cos(alpha_i) - rR*math.cos(alpha_f))\
                + math.sqrt(p2t**2 + (rL + rR)**2)
                
        # Checking if the singular solution is a valid non-degenerate solution
        if p4t >= 10**(-6) and p1t >= 10**(-6) and \
            abs(math.cos(alpha_f)*(-math.sqrt(p2t**2 + (rL + rR)**2) + p4t)\
                - (d + rL*math.sin(alpha_i) + rR*math.sin(alpha_f))) <= 10**(-6) and\
            abs(math.sin(alpha_f)*(-math.sqrt(p2t**2 + (rL + rR)**2) + p4t)\
                - (-rL*math.cos(alpha_i) - rR*math.cos(alpha_f))) <= 10**(-6):
            
            p1_solns_array.append(p1t)
            p2_solns_array.append(p2t)
            p3_solns_array.append(p3t)
            p4_solns_array.append(p4t)
            path_length_array.append((rL*p1t + p2t + rR*p3t + p4t))
            path_cost_array.append(((rL + muL)*p1t + p2t + (rR + muR)*p3t + p4t))
                
    elif path_type.lower() == 'rsls':
            
        # Computing the parameters corresponding to the singular solution,
        # and allocating it to temporary variables
        p3t = math.pi + 2*math.asin(math.sqrt((rL + rR)/(2*(rL + rR + muL + muR))))
        p2t = -(muL + muR)*(1/math.tan(p3t/2))
        p1t = np.mod((alpha_i - alpha_f + p3t), 2*math.pi)
        
        if abs(math.sin(alpha_f)) <= 10**(-2):
            
            # Using the first of the two equations to compute p4
            p4t = (1/math.cos(alpha_f))*(d - rR*math.sin(alpha_i) - rL*math.sin(alpha_f))\
                + math.sqrt(p2t**2 + (rL + rR)**2)
                
        else:
            
            # Using the second of the two equations to compute p4
            p4t = (1/math.sin(alpha_f))*(rR*math.cos(alpha_i) + rL*math.cos(alpha_f))\
                + math.sqrt(p2t**2 + (rL + rR)**2)
                
        # Checking if the singular solution is a valid non-degenerate solution
        if p4t >= 10**(-6) and p1t >= 10**(-6) and \
            abs(math.cos(alpha_f)*(-math.sqrt(p2t**2 + (rL + rR)**2) + p4t)\
                - (d - rR*math.sin(alpha_i) - rL*math.sin(alpha_f))) <= 10**(-6) and\
            abs(math.sin(alpha_f)*(-math.sqrt(p2t**2 + (rL + rR)**2) + p4t)\
                - (rR*math.cos(alpha_i) + rL*math.cos(alpha_f))) <= 10**(-6):
            
            p1_solns_array.append(p1t)
            p2_solns_array.append(p2t)
            p3_solns_array.append(p3t)
            p4_solns_array.append(p4t)
            path_length_array.append((rR*p1t + p2t + rL*p3t + p4t))
            path_cost_array.append(((rR + muR)*p1t + p2t + (rL + muL)*p3t + p4t))
                
    elif path_type.lower() == 'slsr':
            
        # Computing the parameters corresponding to the singular solution,
        # and allocating it to temporary variables
        p2t = math.pi + 2*math.asin(math.sqrt((rL + rR)/(2*(rL + rR + muL + muR))))
        p3t = -(muL + muR)*(1/math.tan(p2t/2))
        p4t = np.mod((alpha_i - alpha_f + p2t), 2*math.pi)
        
        if abs(math.sin(alpha_i)) <= 10**(-2):
            
            # Using the first of the two equations to compute p1
            p1t = (1/math.cos(alpha_i))*(d + rL*math.sin(alpha_i) + rR*math.sin(alpha_f))\
                + math.sqrt(p3t**2 + (rL + rR)**2)
                
        else:
            
            # Using the second of the two equations to compute p1
            p1t = (1/math.sin(alpha_i))*(-rL*math.cos(alpha_i) - rR*math.cos(alpha_f))\
                + math.sqrt(p3t**2 + (rL + rR)**2)
                
        # Checking if the singular solution is a valid non-degenerate solution
        if p1t >= 10**(-6) and p4t >= 10**(-6) and \
            abs(math.cos(alpha_i)*(-math.sqrt(p3t**2 + (rL + rR)**2) + p1t)\
                - (d + rL*math.sin(alpha_i) + rR*math.sin(alpha_f))) <= 10**(-6) and\
            abs(math.sin(alpha_i)*(-math.sqrt(p3t**2 + (rL + rR)**2) + p1t)\
                - (-rL*math.cos(alpha_i) - rR*math.cos(alpha_f))) <= 10**(-6):
            
            p1_solns_array.append(p1t)
            p2_solns_array.append(p2t)
            p3_solns_array.append(p3t)
            p4_solns_array.append(p4t)
            path_length_array.append((p1t + rL*p2t + p3t + rR*p4t))
            path_cost_array.append((p1t + (rL + muL)*p2t + p3t + (rR + muR)*p4t))
                
    elif path_type.lower() == 'srsl':
            
        # Computing the parameters corresponding to the singular solution,
        # and allocating it to temporary variables
        p2t = math.pi + 2*math.asin(math.sqrt((rL + rR)/(2*(rL + rR + muL + muR))))
        p3t = -(muL + muR)*(1/math.tan(p2t/2))
        p4t = np.mod((alpha_f - alpha_i + p2t), 2*math.pi)
        
        if abs(math.sin(alpha_i)) <= 10**(-2):
            
            # Using the first of the two equations to compute p1
            p1t = (1/math.cos(alpha_i))*(d - rR*math.sin(alpha_i) - rL*math.sin(alpha_f))\
                + math.sqrt(p3t**2 + (rL + rR)**2)
                
        else:
            
            # Using the second of the two equations to compute p1
            p1t = (1/math.sin(alpha_i))*(rR*math.cos(alpha_i) + rL*math.cos(alpha_f))\
                + math.sqrt(p3t**2 + (rL + rR)**2)
                
        # Checking if the singular solution is a valid non-degenerate solution
        if p1t >= 10**(-6) and p4t >= 10**(-6) and \
            abs(math.cos(alpha_i)*(-math.sqrt(p3t**2 + (rL + rR)**2) + p1t)\
                - (d - rR*math.sin(alpha_i) - rL*math.sin(alpha_f))) <= 10**(-6) and\
            abs(math.sin(alpha_i)*(-math.sqrt(p3t**2 + (rL + rR)**2) + p1t)\
                - (rR*math.cos(alpha_i) + rL*math.cos(alpha_f))) <= 10**(-6):
            
            p1_solns_array.append(p1t)
            p2_solns_array.append(p2t)
            p3_solns_array.append(p3t)
            p4_solns_array.append(p4t)
            path_length_array.append((p1t + rR*p2t + p3t + rL*p4t))
            path_cost_array.append((p1t + (rR + muR)*p2t + p3t + (rL + muL)*p4t))
                
    # Computing the non-singular solution
    if path_type.lower() == 'lsrs':
        
        c = d*math.sin(alpha_f) + rL*math.cos(alpha_f - alpha_i) + rR
        sin2p3RHS = (rL + rR + 2*(muL + muR) - c)/(2*(rL + rR + muL + muR))
        
        if sin2p3RHS > 0 and sin2p3RHS < 1:
            
            p3t = 2*math.pi - 2*math.asin(math.sqrt(sin2p3RHS))
            p2t = -(muL + muR)*(1/math.tan(p3t/2))
            p1t = np.mod((alpha_f - alpha_i + p3t), 2*math.pi)
            delta = math.atan2((rL + rR), p2t)
            # Computing the solution corresponding to p4
            if abs(math.sin(-p3t + delta)) >= 10**(-6):
                
                p4t = (1/(math.sin(-p3t + delta)))*(-d*math.sin(alpha_f + p3t - delta)\
                                                    -rL*math.cos(p1t - delta)\
                                                    - rR*math.cos(p3t - delta))
                    
                # Checking if the solution set is valid and non-degenerate
                if p4t >= 10**(-6) and p1t >= 10**(-6):
                    
                    p1_solns_array.append(p1t)
                    p2_solns_array.append(p2t)
                    p3_solns_array.append(p3t)
                    p4_solns_array.append(p4t)
                    path_length_array.append((rL*p1t + p2t + rR*p3t + p4t))
                    path_cost_array.append(((rL + muL)*p1t + p2t + (rR + muR)*p3t + p4t))
                    
    elif path_type.lower() == 'rsls':
        
        c = -d*math.sin(alpha_f) + rR*math.cos(alpha_f - alpha_i) + rL
        sin2p3RHS = (rL + rR + 2*(muL + muR) - c)/(2*(rL + rR + muL + muR))
        
        if sin2p3RHS > 0 and sin2p3RHS < 1:
            
            p3t = 2*math.pi - 2*math.asin(math.sqrt(sin2p3RHS))
            p2t = -(muL + muR)*(1/math.tan(p3t/2))
            p1t = np.mod((alpha_i - alpha_f + p3t), 2*math.pi)
            delta = math.atan2((rL + rR), p2t)
            # Computing the solution corresponding to p4
            if abs(math.sin(p3t - delta)) >= 10**(-6):
                
                p4t = (1/(math.sin(p3t - delta)))*(-d*math.sin(alpha_f - p3t + delta)\
                                                   + rR*math.cos(-p1t + delta)\
                                                   + rL*math.cos(-p3t + delta))
                    
                # Checking if the solution set is valid and non-degenerate
                if p4t >= 10**(-6) and p1t >= 10**(-6):
                    
                    p1_solns_array.append(p1t)
                    p2_solns_array.append(p2t)
                    p3_solns_array.append(p3t)
                    p4_solns_array.append(p4t)
                    path_length_array.append((rR*p1t + p2t + rL*p3t + p4t))
                    path_cost_array.append(((rR + muR)*p1t + p2t + (rL + muL)*p3t + p4t))
                    
    elif path_type.lower() == 'slsr':
        
        c = d*math.sin(alpha_i) + rR*math.cos(alpha_f - alpha_i) + rL
        sin2p2RHS = (rL + rR + 2*(muL + muR) - c)/(2*(rL + rR + muL + muR))
        
        if sin2p2RHS > 0 and sin2p2RHS < 1:
            
            p2t = 2*math.pi - 2*math.asin(math.sqrt(sin2p2RHS))
            p3t = -(muL + muR)*(1/math.tan(p2t/2))
            p4t = np.mod((alpha_i - alpha_f + p2t), 2*math.pi)
            delta = math.atan2((rL + rR), p3t)
            # Computing the solution corresponding to p1
            if abs(math.sin(-p2t + delta)) >= 10**(-6):
                
                p1t = (1/(math.sin(-p2t + delta)))*(-d*math.sin(alpha_i + p2t - delta)\
                                                    -rR*math.cos(p4t - delta)\
                                                    - rL*math.cos(p2t - delta))
                    
                # Checking if the solution set is valid and non-degenerate
                if p4t >= 10**(-6) and p1t >= 10**(-6):
                    
                    p1_solns_array.append(p1t)
                    p2_solns_array.append(p2t)
                    p3_solns_array.append(p3t)
                    p4_solns_array.append(p4t)
                    path_length_array.append((p1t + rL*p2t + p3t + rR*p4t))
                    path_cost_array.append((p1t + (rL + muL)*p2t + p3t + (rR + muR)*p4t))
                    
    elif path_type.lower() == 'srsl':
        
        c = -d*math.sin(alpha_i) + rL*math.cos(alpha_f - alpha_i) + rR
        sin2p2RHS = (rL + rR + 2*(muL + muR) - c)/(2*(rL + rR + muL + muR))
        
        if sin2p2RHS > 0 and sin2p2RHS < 1:
            
            p2t = 2*math.pi - 2*math.asin(math.sqrt(sin2p2RHS))
            p3t = -(muL + muR)*(1/math.tan(p2t/2))
            p4t = np.mod((alpha_f - alpha_i + p2t), 2*math.pi)
            delta = math.atan2((rL + rR), p3t)
            # Computing the solution corresponding to p1
            if abs(math.sin(p2t - delta)) >= 10**(-6):
                
                p1t = (1/(math.sin(p2t - delta)))*(-d*math.sin(alpha_i - p2t + delta)\
                                                   + rL*math.cos(-p4t + delta)\
                                                   + rR*math.cos(-p2t + delta))
                    
                # Checking if the solution set is valid and non-degenerate
                if p4t >= 10**(-6) and p1t >= 10**(-6):
                    
                    p1_solns_array.append(p1t)
                    p2_solns_array.append(p2t)
                    p3_solns_array.append(p3t)
                    p4_solns_array.append(p4t)
                    path_length_array.append((p1t + rR*p2t + p3t + rL*p4t))
                    path_cost_array.append((p1t + (rR + muR)*p2t + p3t + (rL + muL)*p4t))
                    
    # Computing the minimum cost solution among the two possible solutions
    if len(path_cost_array) != 0: # If at least one solution exists
            
            # Obtaining the index corresponding to the minimum cost solution
            ind_min_cost_soln = np.nanargmin(path_cost_array)
            path_length = path_length_array[ind_min_cost_soln]
            cost_path = path_cost_array[ind_min_cost_soln]
            p1 = p1_solns_array[ind_min_cost_soln]
            p2 = p2_solns_array[ind_min_cost_soln]
            p3 = p3_solns_array[ind_min_cost_soln]
            p4 = p4_solns_array[ind_min_cost_soln]
    
    return path_length, cost_path, p1, p2, p3, p4

def CSCSC_path(ini_config, fin_config, rL, rR, muL, muR, path_type = 'lsrsl'):
    '''
    This function generates a CSCSC path connecting the initial and final configurations.
    This path could be optimal only for weighted Dubins.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    rL : Scalar
        Radius of the left tight turn.
    rR : Scalar
        Radius of the right tight turn.
    muL : Scalar
        Penalty associated with a left turn.
    muR : Scalar
        Penalty associated with a right turn.
    path_type : String
        Contains the path type, and contains five segments made up of 'l' for left turn,
        'r' for right turn, and 's' for straight line segment. ALlowed path types are
        LSRSL and RSLSR

    Returns
    -------
    path_length : Scalar
        Length of the path.
    cost_path : Scalar
        Cost of the path.
    phi_1 : Scalar
        Angle of the first segment of the path.
    lS : Scalar
        Length of the straight line segments of the path.
    phi_3 : Scalar
        Angle of the middle turn of the path.
    phi_5 : Scalar
        Angle of the final turn of the path

    '''
    
    # Initial and final configurations after modification, i.e., shifting the
    # initial configuration to the origin and the final configuration along
    # the x-axis
    ini_config_mod, fin_config_mod, d = ini_fin_config_manipulate(ini_config, fin_config)
    
    # Initial and final heading angles in the modified frame of reference
    alpha_i = ini_config_mod[2]
    alpha_f = fin_config_mod[2]
    
    path_types = ['lsrsl', 'rslsr']
    
    if path_type.lower() not in path_types:
        
        raise Exception('Incorrect path type is provided')
        
    # Initializing the return variables as np.NaN. If a non-trivial LSRSL path exists,
    # these variables are overwritten.
    path_length = np.NaN
    cost_path = np.NaN
    phi_1 = np.NaN
    lS = np.NaN
    phi_3 = np.NaN
    phi_5 = np.NaN
        
    # Checking if the initial and final configurations lie on the same tight circle
    if abs(alpha_f - (2*math.pi - alpha_i)) <= 10**(-6) and path_type.lower() == 'lsrsl'\
        and abs(d + 2*rL*math.sin(alpha_i)) <= 10**(-6):
            
            print('The LSRSL path is an L path for the given configurations and parameters')
            
    elif abs(alpha_f - (2*math.pi - alpha_i)) <= 10**(-6) and path_type.lower() == 'rslsr'\
        and abs(d - 2*rR*math.sin(alpha_i)) <= 10**(-6):
            
            print('The RSLSR path is an R path for the given configurations and parameters')
            
    else:
        
        # Computing the value of the constant c depending on the path type
        if path_type.lower() == 'lsrsl':
            
            c = 0.5*math.sqrt(d**2 + 2*rL**2 + 2*d*rL*(math.sin(alpha_i) - math.sin(alpha_f))\
                              - 2*(rL**2)*math.cos(alpha_f - alpha_i))
                
        else:
            
            c = 0.5*math.sqrt(d**2 + 2*rR**2 - 2*d*rR*(math.sin(alpha_i) - math.sin(alpha_f))\
                              - 2*(rR**2)*math.cos(alpha_f - alpha_i))
        
        # Obtaining the RHS corresponding to the two possible solutions
        rhs_array = []
        # RHS corresponding to the first solution
        rhs = (c + math.sqrt(c**2 + 4*(rL + rR + muL + muR)*(muL + muR)))/(2*(rL + rR + muL + muR))
        if rhs > 0 and rhs < 1:
            
            rhs_array.append(rhs)
        
        # RHS corresponding to the second solution
        rhs = (-c + math.sqrt(c**2 + 4*(rL + rR + muL + muR)*(muL + muR)))/(2*(rL + rR + muL + muR))       
        if rhs > 0 and rhs < 1:
            
            rhs_array.append(rhs)
            
        # Obtaining the corresponding solutions for phi_3
        phi_3_solns_array = []
        for i in rhs_array:
            
            phi_3_solns_array.append(2*math.pi - 2*math.asin(i))
            
        # Obtaining the corresponding solutions for phi_1, phi_5, and lS, and the
        # cost and length of each path
        phi_1_solns_array = []
        phi_5_solns_array = []
        lS_solns_array = []
        path_length_array = []
        path_cost_array = []
        for i in phi_3_solns_array:
            
            lS_soln = -(muL + muR)*(1/math.tan(i/2))
            lS_solns_array.append(lS_soln)
            
            # Obtaining the sign of the term (rL + rR)*sin(phi_3/2) + s*cos(phi_3/2),
            # since this term is cancelled in the two equations to obtain phi_1.
            # However, the sign of this term should be accounted for while cancelling.
            sign_canc = np.sign((rL + rR)*math.sin(i/2) + lS_soln*math.cos(i/2))
            
            if path_type.lower() == 'lsrsl':
            
                phi_1_soln = np.mod(math.atan2(sign_canc*(-rL*(math.cos(alpha_i) - math.cos(alpha_f))),\
                                               sign_canc*((d + rL*(math.sin(alpha_i) - math.sin(alpha_f)))))\
                                    - alpha_i + i/2, 2*math.pi)
                phi_1_solns_array.append(phi_1_soln)
                phi_5_soln = np.mod(alpha_f - alpha_i - phi_1_soln + i, 2*math.pi)
                phi_5_solns_array.append(phi_5_soln)              
                path_length_array.append(rL*(phi_1_soln + phi_5_soln) + 2*lS_soln + rR*i)
                path_cost_array.append((rL + muL)*(phi_1_soln + phi_5_soln) + 2*lS_soln + (rR + muR)*i)
                
            else: # Computing the parameters for RSLSR paths
            
                phi_1_soln = np.mod(-math.atan2(sign_canc*(rR*(math.cos(alpha_i) - math.cos(alpha_f))),\
                                                sign_canc*((d - rR*(math.sin(alpha_i) - math.sin(alpha_f)))))\
                                    + alpha_i + i/2, 2*math.pi)
                phi_1_solns_array.append(phi_1_soln)
                phi_5_soln = np.mod(alpha_i - alpha_f - phi_1_soln + i, 2*math.pi)
                phi_5_solns_array.append(phi_5_soln)              
                path_length_array.append(rR*(phi_1_soln + phi_5_soln) + 2*lS_soln + rL*i)
                path_cost_array.append((rR + muR)*(phi_1_soln + phi_5_soln) + 2*lS_soln + (rL + muL)*i)
                
        # Choosing the minimum cost solution among the two possible solutions. If
        # only one solution is present, that solution is chosen. If no solutions are
        # present, np.NaN is returned for each variable.
        if len(path_cost_array) != 0: # If at least one solution exists
            
            # Obtaining the index corresponding to the minimum cost solution
            ind_min_cost_soln = np.nanargmin(path_cost_array)
            path_length = path_length_array[ind_min_cost_soln]
            cost_path = path_cost_array[ind_min_cost_soln]
            phi_1 = phi_1_solns_array[ind_min_cost_soln]
            lS = lS_solns_array[ind_min_cost_soln]
            phi_3 = phi_3_solns_array[ind_min_cost_soln]
            phi_5 = phi_5_solns_array[ind_min_cost_soln]
            
        # Checking the existence of a non-degenerate CSCSC path
        if phi_1 <= 10**(-6) or phi_3 <= 10**(-6) or phi_5 <= 10**(-6) or lS <= 10**(-6):
            
            print('The optimal ' + str(path_type.upper()) + ' path is degenerate')
            path_length = np.NaN
            cost_path = np.NaN
            phi_1 = np.NaN
            lS = np.NaN
            phi_3 = np.NaN
            phi_5 = np.NaN
        
    return path_length, cost_path, phi_1, lS, phi_3, phi_5

def dubins_paths(ini_config, fin_config, rL, rR, muL, muR, path_type = 'lsl'):
    '''
    This function generates the appropriate path required for the given configurations
    and parameters using the previously defined functions.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    rL : Scalar
        Radius of the left tight turn.
    rR : Scalar
        Radius of the right tight turn.
    muL : Scalar
        Penalty associated with a left turn.
    muR : Scalar
        Penalty associated with a right turn.
    path_type : String, optional
        Contains the path type to be generated. The default is 'lsl'.

    Returns
    -------
    path_length : Scalar
        Length of the path.
    cost_path : Scalar
        Cost of the path.
    params_path : Numpy array
        Contains the parameters of the path.

    '''
    
    # Listing the path types depending on the class of the paths it belongs to
    path_types_CSC = np.array(['lsl', 'rsr', 'lsr', 'rsl'])
    path_types_CCC = np.array(['lrl', 'rlr'])
    path_types_SCS = np.array(['sls', 'srs'])
    path_types_CSCS_SCSC = np.array(['lsrs', 'rsls', 'slsr', 'srsl'])
    path_types_CSCSC = np.array(['lsrsl', 'rslsr'])
    
    params_path = np.full(5, np.NaN)
    
    # Checking which type of the path has been passed and calling the appropriate function
    if path_type in path_types_CSC:
        
        path_length, cost_path, params_path[0], params_path[1], params_path[2]\
            = CSC_path(ini_config, fin_config, rL, rR, muL, muR, path_type)
            
    elif path_type in path_types_CCC:
        
        path_length, cost_path, params_path[0], params_path[1], params_path[2]\
            = CCC_path(ini_config, fin_config, rL, rR, muL, muR, path_type)
            
    elif path_type in path_types_SCS:
        
        if path_type[1].lower() == 'l':
            
            rturn = rL
            muturn = muL
            
        else:
            
            rturn = rR
            muturn = muR
        
        path_length, cost_path, params_path[0], params_path[1], params_path[2]\
            = SCS_path(ini_config, fin_config, rturn, muturn, path_type)
            
    elif path_type in path_types_CSCS_SCSC:
        
        path_length, cost_path, params_path[0], params_path[1], params_path[2],\
            params_path[3] = SCSC_CSCS_paths(ini_config, fin_config, rL, rR, muL,\
                                             muR, path_type)
    
    elif path_type in path_types_CSCSC:
        
        path_length, cost_path, params_path[0], params_path[1], params_path[2],\
            params_path[4] = CSCSC_path(ini_config, fin_config, rL, rR, muL,\
                                        muR, path_type)
        params_path[3] = params_path[1] # Since both the two straight line segments
        # have the same length
        
    else:
        
        raise Exception('Path type passed is invalid.')
    
    return path_length, cost_path, params_path

def asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, rL, rR,\
                                           muL, muR, filename = 'plots_Dubins_comp.html'):
    '''
    This function generates the paths for asymmetric unweighted and asymmetric weighted
    2D Dubins. If rL = rR, the function can be used for symmetric 2D Dubins.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    rL : Scalar
        Radius of the left tight turn.
    rR : Scalar
        Radius of the right tight turn.
    muL : Scalar
        Penalty associated with a left turn.
    muR : Scalar
        Penalty associated with a right turn.
    filename : String, optional
        Name of the html file in which the plots are generated. The default is
        'plots_Dubins_comp.html'.

    Returns
    -------
    None.

    '''
    
    path_types_unweighted = np.array(['lsl', 'rsr', 'lsr', 'rsl', 'lrl', 'rlr'])
    path_types_weighted = np.array(['lsl', 'rsr', 'lsr', 'rsl', 'lrl', 'rlr', 'sls', 'srs',\
                                    'lsrs', 'rsls', 'slsr', 'srsl', 'lsrsl', 'rslsr'])
    path_lengths_asym_unweighted = np.empty(len(path_types_unweighted))
    path_costs_asym_weighted = np.empty(len(path_types_weighted))
    asym_unweighted_path_params = np.zeros((len(path_types_unweighted), 5))
    asym_weighted_path_params = np.zeros((len(path_types_weighted), 5))
    
    if filename != False:
    
        # Creating an array for writing the lengths and costs of each path onto the html file.
        text_paths_file_asym_unweight = []
        text_paths_file_asym_unweight.append("---------Details of the configurations---------")
        text_paths_file_asym_unweight.append("The initial and final configurations are " + str(ini_config) +\
                                             ", " + str(fin_config))
        text_paths_file_asym_unweight.append("The parameters rL, rR, muL, muR are " +\
                                             str(rL) + ", " + str(rR) + ", " + str(muL)\
                                             + ", " + str(muR))
        text_paths_file_asym_unweight.append("")
        text_paths_file_asym_unweight.append("---------Details of the paths for unweighted 2D Dubins---------")
        text_paths_file_asym_weight = []
        text_paths_file_asym_weight.append("---------Details of the paths for weighted 2D Dubins---------")
    
    # Generating the paths for each type
    for i in range(len(path_types_weighted)):
                
        # Generating the path for asymmetric weighted Dubins
        _, path_costs_asym_weighted[i], asym_weighted_path_params[i] = \
            dubins_paths(ini_config, fin_config, rL, rR, muL, muR,\
                         path_types_weighted[i])
                
        # Adding to the string to be printed if the path exists
        if np.isnan(path_costs_asym_weighted[i]) == False and filename != False:
            
            temp = "Cost of " + path_types_weighted[i].upper() + " path is "\
                    + str(path_costs_asym_weighted[i])
            text_paths_file_asym_weight.append(temp)
            # Appending the parameters corresponding to the path also
            temp = "The parameters of the " + path_types_weighted[i].upper() + \
                " path are " + str(asym_weighted_path_params[i])
            text_paths_file_asym_weight.append(temp)
        
        # Checking if the path needs to be generated for unweighted Dubins also
        if path_types_weighted[i] in path_types_unweighted:
            
            # Generating the path for asymmetric unweighted Dubins
            path_lengths_asym_unweighted[i], _, asym_unweighted_path_params[i] = \
                dubins_paths(ini_config, fin_config, rL, rR, 0, 0, path_types_weighted[i])
                
            # Adding to the string to be printed if the path exists
            if np.isnan(path_lengths_asym_unweighted[i]) == False and filename != False:
                
                temp = "Length of " + path_types_unweighted[i].upper() + " path is "\
                        + str(path_lengths_asym_unweighted[i])
                text_paths_file_asym_unweight.append(temp)
                # Appending the parameters corresponding to the path also
                temp = "The parameters of the " + path_types_unweighted[i].upper() + \
                    " path are " + str(asym_unweighted_path_params[i])
                text_paths_file_asym_unweight.append(temp)
                
    # Finding the optimal path type for each combination        
    opt_path_costs = np.array([min(path_lengths_asym_unweighted), min(path_costs_asym_weighted)])
    opt_path_type_configs = np.array([path_types_weighted[np.nanargmin(path_lengths_asym_unweighted)],\
                                      path_types_weighted[np.nanargmin(path_costs_asym_weighted)]])
        
    print(opt_path_costs)
    print(opt_path_type_configs)
    
    # Obtaining points along the optimal paths for unweighted and weighted Dubins
    # Unweighted Dubins
    path_params_opt_unweighted = asym_unweighted_path_params[np.nanargmin(path_lengths_asym_unweighted)]
    pts_path_x_coords_unweighted, pts_path_y_coords_unweighted\
        = points_path(ini_config, fin_config, rL, rR, path_params_opt_unweighted,\
                      opt_path_type_configs[0])
            
    # Weighted Dubins
    path_params_opt_weighted = asym_weighted_path_params[np.nanargmin(path_costs_asym_weighted)]
    pts_path_x_coords_weighted, pts_path_y_coords_weighted\
        = points_path(ini_config, fin_config, rL, rR, path_params_opt_weighted,\
                      opt_path_type_configs[1])
    
    # Visualizing the paths
    if filename != False:
        
        # Creating a 2D plot. fig_plane is declared as an instant of the class plotting_functions.
        fig_plane = plotting_functions()
        
        # Plotting the initial and final configurations
        fig_plane.points_2D([ini_config[0]], [ini_config[1]], 'red', 'Initial location', 'circle')
        fig_plane.points_2D([fin_config[0]], [fin_config[1]], 'black',\
                            'Final location', 'diamond')
            
        # Adding initial and final headings
        fig_plane.arrows_2D([ini_config[0]], [ini_config[1]], [math.cos(ini_config[2])],\
                            [math.sin(ini_config[2])], 'orange', 'Initial heading', 2)
        fig_plane.arrows_2D([fin_config[0]], [fin_config[1]], [math.cos(fin_config[2])],\
                            [math.sin(fin_config[2])], 'green', 'Final heading', 2)
            
        # Adding labels to the axis and title to the plot
        fig_plane.update_layout_2D('x (m)', [min(ini_config[0], fin_config[0]) - 2*max(rL, rR),\
                                              max(ini_config[0], fin_config[0]) + 2*max(rL, rR)],\
                                    'y (m)', [min(ini_config[1], fin_config[1]) - 2*max(rL, rR),\
                                              max(ini_config[1], fin_config[1]) + 2*max(rL, rR)],\
                                    'Initial and final configurations')
            
        # Writing onto the html file
        fig_plane.writing_fig_to_html(filename, 'w')
        
        # Adding the details of the configurations to the hrml file
        
        # Adding the details of the paths to the html file
        with open(filename, 'a') as f:
            f.write("<br>")
            for i in range(len(text_paths_file_asym_unweight)):
                f.write(text_paths_file_asym_unweight[i] + ".<br />")
        with open(filename, 'a') as f:
            f.write("<br>")
            for i in range(len(text_paths_file_asym_weight)):
                f.write(text_paths_file_asym_weight[i] + ".<br />")
        
        # Creating the plots for the different path types
        for i in range(len(path_types_weighted)):
            
            # Making a copy of the created figure
            fig_plane_path = copy.deepcopy(fig_plane)             
                        
            # Plotting the path of weighted Dubins if the path exists
            if np.isnan(path_costs_asym_weighted[i]) == False:
            
                # Obtaining the coordinates of points along the path using the
                # points path function
                pts_path_x_coords, pts_path_y_coords = points_path(ini_config, fin_config,\
                                                                   rL, rR,\
                                                                   asym_weighted_path_params[i],\
                                                                   path_types_weighted[i])
                    
                # Adding the path onto the figure
                fig_plane_path.scatter_2D(pts_path_x_coords, pts_path_y_coords,\
                                          'blue', 'Path', 3)
                    
                # Adding labels to the axis and title to the plot
                fig_plane_path.update_layout_2D('x (m)', [min(ini_config[0], fin_config[0]) - 3*max(rL, rR),\
                                                      max(ini_config[0], fin_config[0]) + 3*max(rL, rR)],\
                                                'y (m)', [min(ini_config[1], fin_config[1]) - 3*max(rL, rR),\
                                                      max(ini_config[1], fin_config[1]) + 3*max(rL, rR)],\
                                                path_types_weighted[i].upper() + ' path')
                # Writing onto the html file
                fig_plane_path.writing_fig_to_html(filename, 'a')
                    
        # Plotting the optimal path for symmetric unweighted and weighted Dubins
        # and asymmetric unweighted and weighted Dubins
        
        # Making a copy of the created figure
        fig_plane_path = copy.deepcopy(fig_plane)
        text_paths_file_optimal = []
        
        text_paths_file_optimal.append("The optimal path for unweighted Dubins is of type "\
                                       + str(opt_path_type_configs[0].upper()) + \
                                       ', whose cost is ' + str(opt_path_costs[0]) +\
                                       ', and whose parameters are '\
                                       + str(path_params_opt_unweighted))
        # Plotting the optimal path
        fig_plane_path.scatter_2D(pts_path_x_coords_unweighted, pts_path_y_coords_unweighted,\
                                  'red', 'Unweighted 2D Dubins', 3)
        
        text_paths_file_optimal.append("The optimal path for weighted Dubins is of type "\
                                       + str(opt_path_type_configs[1].upper()) + \
                                       ', whose cost is ' + str(opt_path_costs[1]) +\
                                       ', and whose parameters are '\
                                       + str(path_params_opt_weighted))
        # Plotting the optimal path
        fig_plane_path.scatter_2D(pts_path_x_coords_weighted, pts_path_y_coords_weighted,\
                                  'blue', 'Weighted 2D Dubins', 3, 'dash')
                
        # Updating the layout and adding the figure to the html file
        fig_plane_path.update_layout_2D('x (m)', [min(ini_config[0], fin_config[0]) - 3*max(rL, rR),\
                                                  max(ini_config[0], fin_config[0]) + 3*max(rL, rR)],\
                                        'y (m)', [min(ini_config[1], fin_config[1]) - 3*max(rL, rR),\
                                                  max(ini_config[1], fin_config[1]) + 3*max(rL, rR)],\
                                        'Optimal paths for given configurations')
            
        with open(filename, 'a') as f:
            f.write("<br>")
            for i in range(len(text_paths_file_optimal)):
                f.write(text_paths_file_optimal[i] + ".<br />")
                
        # Writing onto the html file
        fig_plane_path.writing_fig_to_html(filename, 'a')
                    
    return path_lengths_asym_unweighted, path_costs_asym_weighted, opt_path_type_configs, pts_path_x_coords_unweighted,\
        pts_path_y_coords_unweighted, pts_path_x_coords_weighted, pts_path_y_coords_weighted

def parametric_study_weighted_Dubins_paths(alpha_i_range, d_range, alpha_f_range,\
                                           rR_range, muL_range, muR_range):
    '''
    In this function, a parametric study for the optimal path for weighted Dubins
    is conducted. Without loss of generality, rL is set to one. The initial position
    is set at the origin and the final position is set along the x-axis without
    loss of generality.

    Parameters
    ----------
    alpha_i_range : Numpy array
        Range of the initial heading angles.
    d_range : Numpy array
        Range of the distance of the final position along the x-axis.
    alpha_f_range : Numpy array
        Range of the final heading angles.
    rR_range : Numpy array
        Range of the radius of the right turn.
    muL_range : Numpy array
        Range of the penalty corresponding to the left turn.
    muR_range : Numpy array
        Range of the penalty corresponding to the right turn.

    Returns
    -------
    None.

    '''
    
    # Declaring the type of paths to be generated
    path_types_weighted = np.array(['lsl', 'rsr', 'lsr', 'rsl', 'lrl', 'rlr', 'sls', 'srs',\
                                    'lsrs', 'rsls', 'slsr', 'srsl', 'lsrsl', 'rslsr'])
    path_types_optimal_path_count = np.zeros(np.shape(path_types_weighted))
    
    for i in alpha_i_range:
        for j in d_range:
            for k in alpha_f_range:
                for l in rR_range:
                    for m in muL_range:
                        for n in muR_range:
                            
                            # Generating the initial and final configuration arrays
                            ini_config = np.array([0, 0, i])
                            fin_config = np.array([j, 0, k])
                            
                            if np.linalg.norm(ini_config - fin_config) <= 10**(-3):
                                
                                continue
                            
                            # Obtaining the optimal path type for unweighted and
                            # weighted Dubins using the asymmetric_weight_2D_Dubins_comparison
                            # function
                            
                            _, _, opt_path_type, _, _, _, _ = \
                                asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config,\
                                                                       1, l, m, n, False)
                            # Incrementing the counter corresponding to the optimal
                            # path for the weighted Dubins
                            opt_path_ind =\
                                np.where(path_types_weighted == opt_path_type[1].lower())[0][0]
                            path_types_optimal_path_count[opt_path_ind] += 1
                            
                            if opt_path_type[1].lower() == 'lrl' or\
                                opt_path_type[1].lower() == 'rlr':
                                    
                                    print('The optimal path type is ' +\
                                          str(opt_path_type[1].upper()))
                                    print('alpha_i = ' + str(i) + ', d = ' + str(j) + \
                                          ', alpha_f = ' + str(k) + ',  rR = ' + str(l) +\
                                          ', muL = ' + str(m) + ', muR = ' + str(n))
                                    exit()
                            
    # Plotting the number of times a path is optimal
    # Plotting the comparison
    x = [i for i in range(len(path_types_weighted))]    
    labels = [str(i.upper()) for i in path_types_weighted]
    plt.plot(path_types_optimal_path_count, '*', linewidth = 1)
    plt.xlabel('Path type')
    plt.ylabel('Number of configurations')
    plt.xticks(x, labels)
    plt.show()
    
    return path_types_weighted, path_types_optimal_path_count

def parametric_study_weighted_Dubins_paths_fixed_config(ini_config, fin_config,\
                                                        rR_range, muL, muR_range):
    '''
    In this function, a parametric study for the optimal path for weighted Dubins
    is conducted. Without loss of generality, rL is set to one. The initial and
    final configurations are fixed in this study, and rR and muR are varied.

    Parameters
    ----------
    ini_config : Numpy array
        Contains the initial position and heading angle.
    fin_config : Numpy array
        Contains the final position and heading angle.
    rR_range : Numpy array
        Range of the radius of the right turn.
    muL : Scalar
        Fixed penalty for the left turn.
    muR_range : Numpy array
        Range of the penalty corresponding to the right turn.

    Returns
    -------
    None.

    '''
    
    # Arrays for path types for unweighted and weighted. NOTE THAT THE SEQUENCE OF
    # PATHS USED IN PATH_TYPES_WEIGHTED IS THE SAME AS THAT IN THE FUNCTION
    # asymmetric_weight_2D_Dubins_comparison
    path_types_unweighted = np.array(['lsl', 'rsr', 'lsr', 'rsl', 'lrl', 'rlr'])
    path_types_weighted = np.array(['lsl', 'rsr', 'lsr', 'rsl', 'lrl', 'rlr', 'sls', 'srs',\
                                    'lsrs', 'rsls', 'slsr', 'srsl', 'lsrsl', 'rslsr'])
    
    # Generating a matrix to store the type of paths for each discretization
    path_types_optimal = np.empty((len(rR_range), len(muR_range)), dtype = object)
    path_cost_diff = np.empty((len(rR_range), len(muR_range)), dtype = object)
    
    for l in range(len(rR_range)):
        for n in range(len(muR_range)):
            
            # Obtaining the optimal path type for unweighted and
            # weighted Dubins using the asymmetric_weight_2D_Dubins_comparison
            # function
            print(rR_range[l])
            print(muR_range[n])
            
            # Obtaining the optimal path for weighted Dubins
            _, path_cost_weighted, opt_path_type, _, _, _, _ = \
                asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config,\
                                                       1, rR_range[l], muL,\
                                                       muR_range[n], False)
            
            print(opt_path_type[1])
            
            # Storing the optimal path type in the matrix
            path_types_optimal[l, n] = opt_path_type[1].lower()
            # Storing the difference in cost between weighted and unweighted
            print(path_cost_weighted)
            
            # Obtaining the minimum cost path corresponding to the CSC and CCC path types
            min_cost_path_unweighted = np.infty
            for i in range(len(path_types_weighted)):
                
                if (path_types_weighted[i] in path_types_unweighted) and\
                    (path_cost_weighted[i] < min_cost_path_unweighted):
                        
                    min_cost_path_unweighted = path_cost_weighted[i]
            
            # Returning the percentage deviation from optimum
            path_cost_diff[l, n] = (min_cost_path_unweighted - min(path_cost_weighted))*100/min(path_cost_weighted)
            
            if opt_path_type[1].lower() == 'lrl' or\
                opt_path_type[1].lower() == 'rlr':
                    
                    print('The optimal path type is ' +\
                          str(opt_path_type[1].upper()))
                    print('rR = ' + str(rR_range[l]) + ', muR = ' + str(muR_range[n]))
                    # exit()
    
    return path_types_optimal, path_cost_diff

def parametric_study_weighted_Dubins_paths_fixed_param_vary_config\
    (pos_x_var, pos_y_var, fin_heading, rL, rR, muL, muR):
    '''
    In this function, a parametric study is performed for fixed parameters and
    varying location in the XY-plane. Note that in this study, the final heading
    angle is fixed. Moreover, in this study, the initial location is at the origin
    and the initial heading angle is 0 rad.

    Parameters
    ----------
    pos_x_var : Array
        Contains the variation in the X-axis.
    pos_y_var : Array
        Contains the variation in the Y-axis.
    fin_heading : Scalar
        Heading angle at the final configuration in radians.
    rL : Scalar
        Radius of left turn.
    rR : Scalar
        Radius of right turn.
    muL : Scalar
        Penalty of left turn.
    muR : Scalar
        Penalty of right turn.

    Returns
    -------
    path_types_optimal : Matrix
        Contains the optimal path type for every final location.
    path_cost : Matrix
        Contains the cost of the optimal path for every final location.

    '''
    
    ini_config = np.array([0, 0, 0])
    
    # Declaring matrices to store the optimal path type and path cost
    path_types_optimal = np.empty((len(pos_y_var), len(pos_x_var)), dtype = object)
    path_cost = np.empty((len(pos_y_var), len(pos_x_var)), dtype = object)
    
    for i in range(len(pos_y_var)):
        for j in range(len(pos_x_var)):
            
            # Declaring the final configuration
            fin_config = np.array([pos_x_var[j], pos_y_var[i], fin_heading])
            # Obtaining the optimal path for weighted Dubins
            _, path_cost_weighted, opt_path_type, _, _, _, _ = \
                asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config,\
                                                       rL, rR, muL, muR, False)
            # Storing the solution
            path_types_optimal[i, j] = opt_path_type[1].lower()
            path_cost[i, j] = path_cost_weighted
    
    return path_types_optimal, path_cost

# def asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, r_sym, mu_sym, rL,\
#                                            rR, muL, muR, filename = 'plots_Dubins_comp.html'):
#     '''
#     This function generates the paths for symmetric unweighted 2D Dubins, symmetric
#     weighted 2D Dubins, asymmetric unweighted 2D Dubins, and asymmetric weighted
#     2D Dubins.

#     Parameters
#     ----------
#     ini_config : Numpy 1x3 array
#         Contains the initial configuration.
#     fin_config : Numpy 1x3 array
#         Contains the final configuration.
#     r_sym : Scalar
#         Radius of tight turn for symmetric 2D Dubins.
#     mu_sym : Scalar
#         Penalty associated with a turn for symmetric 2D Dubins.
#     rL : Scalar
#         Radius of the left tight turn.
#     rR : Scalar
#         Radius of the right tight turn.
#     muL : Scalar
#         Penalty associated with a left turn.
#     muR : Scalar
#         Penalty associated with a right turn.
#     filename : String, optional
#         Name of the html file in which the plots are generated. The default is
#         'plots_Dubins_comp.html'.

#     Returns
#     -------
#     None.

#     '''
    
#     path_types_unweighted = np.array(['lsl', 'rsr', 'lsr', 'rsl', 'lrl', 'rlr'])
#     path_types_weighted = np.array(['lsl', 'rsr', 'lsr', 'rsl', 'lrl', 'rlr', 'sls', 'srs',\
#                                     'lsrsl', 'rslsr'])
#     path_lengths_sym_unweighted = np.empty(len(path_types_unweighted))
#     path_costs_sym_weighted = np.empty(len(path_types_weighted))
#     path_lengths_asym_unweighted = np.empty(len(path_types_unweighted))
#     path_costs_asym_weighted = np.empty(len(path_types_weighted))
#     sym_unweighted_path_params = np.zeros((len(path_types_unweighted), 5))
#     sym_weighted_path_params = np.zeros((len(path_types_weighted), 5))
#     asym_unweighted_path_params = np.zeros((len(path_types_unweighted), 5))
#     asym_weighted_path_params = np.zeros((len(path_types_weighted), 5))
    
#     if filename != False:
    
#         # Creating an array for writing the lengths and costs of each path onto the html file.
#         text_paths_file_sym_unweight = []
#         text_paths_file_sym_unweight.append("---------Details of the paths for symmetric unweighted Dubins---------")
#         text_paths_file_sym_weight = []
#         text_paths_file_sym_weight.append("---------Details of the paths for symmetric weighted Dubins---------")
#         text_paths_file_asym_unweight = []
#         text_paths_file_asym_unweight.append("---------Details of the paths for asymmetric unweighted Dubins---------")
#         text_paths_file_asym_weight = []
#         text_paths_file_asym_weight.append("---------Details of the paths for asymmetric weighted Dubins---------")
    
#     # Generating the paths for each type
#     for i in range(len(path_types_weighted)):
        
#         # Generating the path for symmetric weighted Dubins
#         _, path_costs_sym_weighted[i], sym_weighted_path_params[i] = \
#             dubins_paths(ini_config, fin_config, r_sym, r_sym, mu_sym, mu_sym,\
#                          path_types_weighted[i])
                
#         # Appending the results to the corresponding string for printing if
#         # the path exists for symmetric weighted Dubins
#         if np.isnan(path_costs_sym_weighted[i]) == False and filename != False:
            
#             temp = "Cost of " + path_types_weighted[i].upper() + " path is "\
#                 + str(path_costs_sym_weighted[i])
#             text_paths_file_sym_weight.append(temp)
#             # Appending the parameters corresponding to the path also
#             temp = "The parameters of the " + path_types_weighted[i].upper() + \
#                 " path are " + str(sym_weighted_path_params[i])
#             text_paths_file_sym_weight.append(temp)
                
#         # Generating the path for asymmetric weighted Dubins
#         _, path_costs_asym_weighted[i], asym_weighted_path_params[i] = \
#             dubins_paths(ini_config, fin_config, rL, rR, muL, muR,\
#                          path_types_weighted[i])
        
#         # Checking if the path needs to be generated for unweighted Dubins also
#         if path_types_weighted[i] in path_types_unweighted:
            
#             # Generating the path for symmetric unweighted Dubins
#             path_lengths_sym_unweighted[i], _, sym_unweighted_path_params[i] = \
#                 dubins_paths(ini_config, fin_config, r_sym, r_sym, 0, 0,\
#                              path_types_weighted[i])
#             # Generating the path for asymmetric unweighted Dubins
#             path_lengths_asym_unweighted[i], _, asym_unweighted_path_params[i] = \
#                 dubins_paths(ini_config, fin_config, rL, rR, 0, 0, path_types_weighted[i])
                
#     # Visualizing the paths
#     if filename != False:
        
#         # Creating a 2D plot. fig_plane is declared as an instant of the class plotting_functions.
#         fig_plane = plotting_functions()
        
#         # Plotting the initial and final configurations
#         fig_plane.points_2D([ini_config[0]], [ini_config[1]], 'red', 'Initial location', 'circle')
#         fig_plane.points_2D([fin_config[0]], [fin_config[1]], 'black',\
#                             'Final location', 'diamond')
            
#         # Adding initial and final headings
#         fig_plane.arrows_2D([ini_config[0]], [ini_config[1]], [math.cos(ini_config[2])],\
#                             [math.sin(ini_config[2])], 'orange', 'Initial heading', 2)
#         fig_plane.arrows_2D([fin_config[0]], [fin_config[1]], [math.cos(fin_config[2])],\
#                             [math.sin(fin_config[2])], 'green', 'Final heading', 2)
            
#         # Adding labels to the axis and title to the plot
#         fig_plane.update_layout_2D('x (m)', [min(ini_config[0], fin_config[0]) - 2*max(rL, rR),\
#                                               max(ini_config[0], fin_config[0]) + 2*max(rL, rR)],\
#                                     'y (m)', [min(ini_config[1], fin_config[1]) - 2*max(rL, rR),\
#                                               max(ini_config[1], fin_config[1]) + 2*max(rL, rR)],\
#                                     'Initial and final configurations')
            
#         # Writing onto the html file
#         fig_plane.writing_fig_to_html(filename, 'w')
        
#         # Creating the plots for the different path types
#         for i in range(len(path_types_weighted)):
            
#             # Making a copy of the created figure
#             fig_plane_path = copy.deepcopy(fig_plane)             
                
#             # Plotting the path of symmetric weighted Dubins if the path exists
#             if np.isnan(path_costs_sym_weighted[i]) == False:
            
#                 # Obtaining the coordinates of points along the path using the
#                 # points path function
#                 pts_path_x_coords, pts_path_y_coords = points_path(ini_config, fin_config,\
#                                                                     r_sym, r_sym,\
#                                                                     sym_weighted_path_params[i],\
#                                                                     path_types_weighted[i])
                    
#                 # Adding the path onto the figure
#                 fig_plane_path.scatter_2D(pts_path_x_coords, pts_path_y_coords,\
#                                           'blue', 'Symmetric 2D Dubins')
                        
#             # Plotting the path of asymmetric weighted Dubins if the path exists
#             if np.isnan(path_costs_sym_weighted[i]) == False:
            
#                 # Obtaining the coordinates of points along the path using the
#                 # points path function
#                 pts_path_x_coords, pts_path_y_coords = points_path(ini_config, fin_config,\
#                                                                    rL, rR,\
#                                                                    asym_weighted_path_params[i],\
#                                                                    path_types_weighted[i])
                    
#                 # Adding the path onto the figure
#                 fig_plane_path.scatter_2D(pts_path_x_coords, pts_path_y_coords,\
#                                           'red', 'Asymmetric 2D Dubins', 3, 'dash')
            
#             # Appending the figure to the html file if either symmetric or asymmetric
#             # weighted Dubins path exists
#             if np.isnan(path_costs_sym_weighted[i]) == False or \
#                 np.isnan(path_costs_asym_weighted[i]) == False:
                    
#                     # Adding labels to the axis and title to the plot
#                     fig_plane_path.update_layout_2D('x (m)', [min(ini_config[0], fin_config[0]) - 3*max(rL, rR),\
#                                                           max(ini_config[0], fin_config[0]) + 3*max(rL, rR)],\
#                                                     'y (m)', [min(ini_config[1], fin_config[1]) - 3*max(rL, rR),\
#                                                           max(ini_config[1], fin_config[1]) + 3*max(rL, rR)],\
#                                                     path_types_weighted[i].upper() + ' path')
#                     # Writing onto the html file
#                     fig_plane_path.writing_fig_to_html(filename, 'a')
                    
#         # Plotting the optimal path for symmetric unweighted and weighted Dubins
#         # and asymmetric unweighted and weighted Dubins
#         # Finding the optimal path type for each combination        
#         opt_path_costs = np.array([min(path_lengths_sym_unweighted), min(path_costs_sym_weighted),\
#                                     min(path_lengths_asym_unweighted), min(path_costs_asym_weighted)])
#         opt_path_type_configs = np.array([path_types_weighted[np.nanargmin(path_lengths_sym_unweighted)],\
#                                           path_types_weighted[np.nanargmin(path_costs_sym_weighted)],\
#                                           path_types_weighted[np.nanargmin(path_lengths_asym_unweighted)],\
#                                           path_types_weighted[np.nanargmin(path_costs_asym_weighted)]])
#         opt_path_configs = np.array(["Symmetric unweighted 2D Dubins", "Symmetric weighted 2D Dubins",\
#                                     "Asymmetric unweighted 2D Dubins", "Asymmetric weighted 2D Dubins"])
#         plot_styles = np.array(['solid', 'dash', 'dashdot', 'dot'])
#         plot_colors = np.array(['red', 'blue', 'green', 'brown'])
        
#         print(opt_path_costs)
#         print(opt_path_type_configs)
        
#         # Making a copy of the created figure
#         fig_plane_path = copy.deepcopy(fig_plane)
        
#         # Creating the optimal paths
#         for i in range(len(opt_path_costs)):
            
#             # Deciding the values to be passed depending on the type of configuration plotted
#             # In addition, the parameters of the optimal path are also obtained
#             if opt_path_configs[i] == "Symmetric unweighted 2D Dubins":
                
#                 rL_opt = r_sym
#                 rR_opt = r_sym
#                 params_opt = sym_unweighted_path_params[np.nanargmin(path_lengths_sym_unweighted)]
                
#             elif opt_path_configs[i] == "Symmetric weighted 2D Dubins":
                
#                 rL_opt = r_sym
#                 rR_opt = r_sym
#                 params_opt = sym_weighted_path_params[np.nanargmin(path_costs_sym_weighted)]
                
#             elif opt_path_configs[i] == "Asymmetric unweighted 2D Dubins":
                
#                 rL_opt = rL
#                 rR_opt = rR
#                 params_opt = asym_unweighted_path_params[np.nanargmin(path_lengths_asym_unweighted)]
                
#             elif opt_path_configs[i] == "Asymmetric weighted 2D Dubins":
                
#                 rL_opt = rL
#                 rR_opt = rR
#                 params_opt = asym_weighted_path_params[np.nanargmin(path_costs_asym_weighted)]
                    
#             # Obtaining the coordinates of points along the path
#             pts_path_x_coords, pts_path_y_coords = points_path(ini_config, fin_config,\
#                                                                rL_opt, rR_opt, params_opt,\
#                                                                opt_path_type_configs[i])
                
#             # Plotting the optimal path
#             fig_plane_path.scatter_2D(pts_path_x_coords, pts_path_y_coords,\
#                                       plot_colors[i], opt_path_configs[i], 3, plot_styles[i])
                
#         # Updating the layout and adding the figure to the html file
#         fig_plane_path.update_layout_2D('x (m)', [min(ini_config[0], fin_config[0]) - 3*max(rL, rR),\
#                                                   max(ini_config[0], fin_config[0]) + 3*max(rL, rR)],\
#                                         'y (m)', [min(ini_config[1], fin_config[1]) - 3*max(rL, rR),\
#                                                   max(ini_config[1], fin_config[1]) + 3*max(rL, rR)],\
#                                         'Optimal paths for given configurations')
#         # Writing onto the html file
#         fig_plane_path.writing_fig_to_html(filename, 'a')
                    
#     return path_lengths_sym_unweighted, path_costs_sym_weighted, path_lengths_asym_unweighted,\
#         path_costs_asym_weighted

# def asymmetric_weight_2D_Dubins_comparison(ini_config, fin_config, r_sym, mu_sym, rL,\
#                                            rR, muL, muR, filename = 'plots_Dubins_comp.html'):
#     '''
#     This function generates the paths for symmetric unweighted 2D Dubins, symmetric
#     weighted 2D Dubins, asymmetric unweighted 2D Dubins, and asymmetric weighted
#     2D Dubins.

#     Parameters
#     ----------
#     ini_config : Numpy 1x3 array
#         Contains the initial configuration.
#     fin_config : Numpy 1x3 array
#         Contains the final configuration.
#     r_sym : Scalar
#         Radius of tight turn for symmetric 2D Dubins.
#     mu_sym : Scalar
#         Penalty associated with a turn for symmetric 2D Dubins.
#     rL : Scalar
#         Radius of the left tight turn.
#     rR : Scalar
#         Radius of the right tight turn.
#     muL : Scalar
#         Penalty associated with a left turn.
#     muR : Scalar
#         Penalty associated with a right turn.
#     filename : String, optional
#         Name of the html file in which the plots are generated. The default is
#         'plots_Dubins_comp.html'.

#     Returns
#     -------
#     None.

#     '''
    
#     path_types = np.array(['lsl', 'rsr', 'lsr', 'rsl', 'lrl', 'rlr'])
#     path_lengths_sym_unweighted = np.empty(6)
#     path_costs_sym_weighted = np.empty(6)
#     path_lengths_asym_unweighted = np.empty(6)
#     path_costs_asym_weighted = np.empty(6)
#     sym_path_params = np.empty((6, 3))
#     asym_path_params = np.empty((6, 3))
    
#     if filename != False:
    
#         # Creating an array for writing the lengths and costs of each path onto the html file.
#         text_paths_file = []
#         text_paths_file.append("---------Details of the paths for symmetric Dubins---------")
#         text_paths_file_asym = []
#         text_paths_file_asym.append("---------Details of the paths for asymmetric Dubins---------")
    
#     # Generating the paths for each type
#     for i in range(len(path_types)):
        
#         # Checking if path is CSC type
#         if path_types[i][1] == 's':
            
#             # Obtaining the path length and costs
#             # path_lengths_sym_unweighted[i] = CSC_path(ini_config, fin_config, r_sym,\
#             #                                           r_sym, 0, 0, path_types[i])[0]
#             path_lengths_sym_unweighted[i], path_costs_sym_weighted[i], sym_path_params[i][0],\
#                 sym_path_params[i][1], sym_path_params[i][2] = CSC_path(ini_config, fin_config,\
#                                                                         r_sym, r_sym, mu_sym,\
#                                                                         mu_sym, path_types[i])
#             # path_lengths_asym_unweighted[i] = CSC_path(ini_config, fin_config, rL,\
#             #                                            rR, 0, 0, path_types[i])[0]
#             path_lengths_asym_unweighted[i], path_costs_asym_weighted[i], asym_path_params[i][0],\
#                 asym_path_params[i][1], asym_path_params[i][2] = CSC_path(ini_config, fin_config,\
#                                                                           rL, rR, muL, muR,\
#                                                                           path_types[i])
                
#         # Checking if path is CCC type
#         else:
            
#             # Obtaining the path length and costs
#             # path_lengths_sym_unweighted[i] = CCC_path(ini_config, fin_config, r_sym,\
#             #                                           r_sym, 0, 0, path_types[i])[0]
#             path_lengths_sym_unweighted[i], path_costs_sym_weighted[i], sym_path_params[i][0],\
#                 sym_path_params[i][1], sym_path_params[i][2] = CCC_path(ini_config, fin_config,\
#                                                                         r_sym, r_sym, mu_sym,\
#                                                                         mu_sym, path_types[i])
#             # path_lengths_asym_unweighted[i] = CCC_path(ini_config, fin_config, rL,\
#             #                                            rR, 0, 0, path_types[i])[0]
#             path_lengths_asym_unweighted[i], path_costs_asym_weighted[i], asym_path_params[i][0],\
#                 asym_path_params[i][1], asym_path_params[i][2] = CCC_path(ini_config, fin_config,\
#                                                                           rL, rR, muL, muR,\
#                                                                           path_types[i])
                
#         if np.isnan(path_lengths_sym_unweighted[i]) == False and filename != False: 
#         # Checking if path exists for symmetric Dubins
            
#             # Appending the details of the path onto the text array
#             temp = "Length of " + path_types[i].upper() + " path for symmetric Dubins is "\
#                 + str(path_lengths_sym_unweighted[i])
#             text_paths_file.append(temp)
#             temp = "Cost of " + path_types[i].upper() + " path for symmetric Dubins with weight is "\
#                 + str(path_costs_sym_weighted[i])
#             text_paths_file.append(temp)
#             # Appending the parameters corresponding to the path also
#             temp = "The three parameters of the " + path_types[i].upper() + \
#                 " path for symmetric Dubins are " + str(sym_path_params[i])
#             text_paths_file.append(temp)
            
#         if np.isnan(path_lengths_asym_unweighted[i]) == False and filename != False: 
#         # Checking if path exists for asymmetric Dubins
            
#             temp = "Length of " + path_types[i].upper() + " path for asymmetric Dubins is "\
#                 + str(path_lengths_asym_unweighted[i])
#             text_paths_file_asym.append(temp)
#             temp = "Cost of " + path_types[i].upper() + " path for asymmetric Dubins with weights is "\
#                 + str(path_costs_asym_weighted[i])
#             text_paths_file_asym.append(temp)
#             # Appending the parameters corresponding to the path also
#             temp = "The three parameters of the " + path_types[i].upper() + \
#                 " path for asymmetric Dubins are " + str(asym_path_params[i])
#             text_paths_file_asym.append(temp)
                
#     # Plotting the paths
#     if filename != False:
        
#         # Creating a 2D plot. fig_plane is declared as an instant of the class plotting_functions.
#         fig_plane = plotting_functions()
        
#         # Plotting the initial and final configurations
#         fig_plane.points_2D([ini_config[0]], [ini_config[1]], 'red', 'Initial location', 'circle')
#         fig_plane.points_2D([fin_config[0]], [fin_config[1]], 'black',\
#                             'Final location', 'diamond')
            
#         # Adding initial and final headings
#         fig_plane.arrows_2D([ini_config[0]], [ini_config[1]], [math.cos(ini_config[2])],\
#                             [math.sin(ini_config[2])], 'orange', 'Initial heading', 2)
#         fig_plane.arrows_2D([fin_config[0]], [fin_config[1]], [math.cos(fin_config[2])],\
#                             [math.sin(fin_config[2])], 'green', 'Final heading', 2)
            
#         # Adding labels to the axis and title to the plot
#         fig_plane.update_layout_2D('x (m)', [min(ini_config[0], fin_config[0]) - 2*max(rL, rR),\
#                                              max(ini_config[0], fin_config[0]) + 2*max(rL, rR)],\
#                                    'y (m)', [min(ini_config[1], fin_config[1]) - 2*max(rL, rR),\
#                                              max(ini_config[1], fin_config[1]) + 2*max(rL, rR)],\
#                                    'Initial and final configurations')
            
#         # Writing onto the html file
#         fig_plane.writing_fig_to_html(filename, 'w')
        
#         # Adding the details of the paths to the html file
#         with open(filename, 'a') as f:
#             f.write("<br>")
#             for i in range(len(text_paths_file)):
#                 f.write(text_paths_file[i] + ".<br />")
#         with open(filename, 'a') as f:
#             f.write("<br>")
#             for i in range(len(text_paths_file_asym)):
#                 f.write(text_paths_file_asym[i] + ".<br />")
        
#         # Creating the plots for the different path types
#         for i in range(len(path_types)):
            
#             # Making a copy of the created figure
#             fig_plane_path = copy.deepcopy(fig_plane)
            
#             # Adding the plot for the symmetric Dubins if the path exists
#             if np.isnan(path_lengths_sym_unweighted[i]) == False:
                
#                 # if path_types[i][1] == 's':
                
#                 #     # Obtaining the parameters of the path for the unweighted Dubins.
#                 #     _, _, phi_1, param_seg_2, phi_3 = CSC_path(ini_config, fin_config,\
#                 #                                                r_sym, r_sym, 0, 0, path_types[i])
                        
                        
#                 # else:
                    
#                 #     # Obtaining the parameters of the path for the unweighted Dubins.
#                 #     _, _, phi_1, param_seg_2, phi_3 = CCC_path(ini_config, fin_config, r_sym,\
#                 #                                                r_sym, 0, 0, path_types[i])
                
#                 # Obtaining the coordinates of points along the path using the
#                 # points path function
#                 pts_path_x_coords, pts_path_y_coords = points_path(ini_config, fin_config,\
#                                                                    r_sym, r_sym, sym_path_params[i],\
#                                                                    path_types[i])
                    
#                 # Adding the path onto the figure
#                 fig_plane_path.scatter_2D(pts_path_x_coords, pts_path_y_coords,\
#                                           'blue', 'Symmetric 2D Dubins')
                    
#             # Adding the plot for the asymmetric Dubins if the path exists
#             if np.isnan(path_lengths_asym_unweighted[i]) == False:
                
#                 # if path_types[i][1] == 's':
                
#                 #     # Obtaining the parameters of the path for the unweighted Dubins.
#                 #     _, _, phi_1, param_seg_2, phi_3 = CSC_path(ini_config, fin_config,\
#                 #                                                rL, rR, 0, 0, path_types[i])
                        
                        
#                 # else:
                    
#                 #     # Obtaining the parameters of the path for the unweighted Dubins.
#                 #     _, _, phi_1, param_seg_2, phi_3 = CCC_path(ini_config, fin_config, rL,\
#                 #                                                rR, 0, 0, path_types[i])
                
#                 # Obtaining the coordinates of points along the path using the
#                 # points path function
#                 pts_path_x_coords, pts_path_y_coords = points_path(ini_config, fin_config,\
#                                                                    rL, rR, asym_path_params[i],\
#                                                                    path_types[i])
                    
#                 # Adding the path onto the figure
#                 fig_plane_path.scatter_2D(pts_path_x_coords, pts_path_y_coords,\
#                                           'red', 'Asymmetric 2D Dubins', 3, 'dash')
            
#             # Appending the figure to the html file if either symmetric or asymmetric
#             # Dubins path exists
#             if np.isnan(path_lengths_sym_unweighted[i]) == False or \
#                 np.isnan(path_lengths_asym_unweighted[i]) == False:
                    
#                     # Adding labels to the axis and title to the plot
#                     fig_plane_path.update_layout_2D('x (m)', [min(ini_config[0], fin_config[0]) - 3*max(rL, rR),\
#                                                          max(ini_config[0], fin_config[0]) + 3*max(rL, rR)],\
#                                                     'y (m)', [min(ini_config[1], fin_config[1]) - 3*max(rL, rR),\
#                                                          max(ini_config[1], fin_config[1]) + 3*max(rL, rR)],\
#                                                     path_types[i].upper() + ' path')
#                     # Writing onto the html file
#                     fig_plane_path.writing_fig_to_html(filename, 'a')
                    
#         # Plotting the optimal paths for each combination
#         # Finding the optimal path type for each combination        
#         opt_path_costs = np.array([min(path_lengths_sym_unweighted), min(path_costs_sym_weighted),\
#                                    min(path_lengths_asym_unweighted), min(path_costs_asym_weighted)])
#         opt_path_type_configs = np.array([path_types[np.nanargmin(path_lengths_sym_unweighted)],\
#                                          path_types[np.nanargmin(path_costs_sym_weighted)],\
#                                          path_types[np.nanargmin(path_lengths_asym_unweighted)],\
#                                          path_types[np.nanargmin(path_costs_asym_weighted)]])
#         opt_path_configs = np.array(["Symmetric unweighted Dubins", "Symmetric weighted Dubins",\
#                                     "Asymmetric unweighted Dubins", "Asymmetric weighted Dubins"])
#         plot_styles = np.array(['solid', 'dash', 'dashdot', 'dot'])
#         plot_colors = np.array(['red', 'blue', 'green', 'brown'])
        
#         print(opt_path_costs)
#         print(opt_path_type_configs)
        
#         # Making a copy of the created figure
#         fig_plane_path = copy.deepcopy(fig_plane)
        
#         # Creating the optimal paths
#         for i in range(len(opt_path_costs)):
            
#             # Deciding the values to be passed depending on the type of configuration plotted
#             if opt_path_configs[i] == "Symmetric unweighted Dubins":
                
#                 muL_opt = 0
#                 muR_opt = 0
#                 rL_opt = r_sym
#                 rR_opt = r_sym
                
#             elif opt_path_configs[i] == "Symmetric weighted Dubins":
                
#                 muL_opt = mu_sym
#                 muR_opt = mu_sym
#                 rL_opt = r_sym
#                 rR_opt = r_sym
                
#             elif opt_path_configs[i] == "Asymmetric unweighted Dubins":
                
#                 muL_opt = 0
#                 muR_opt = 0
#                 rL_opt = rL
#                 rR_opt = rR
                
#             elif opt_path_configs[i] == "Asymmetric weighted Dubins":
                
#                 muL_opt = muL
#                 muR_opt = muR
#                 rL_opt = rL
#                 rR_opt = rR
                
#             # Obtainng the parameters of the optimal path for each configuration
#             if opt_path_type_configs[i][1].lower() == 's':
                
#                 _, _, phi_1, param_seg_2, phi_3 = CSC_path(ini_config, fin_config,\
#                                                            rL_opt, rR_opt, muL_opt, muR_opt,\
#                                                            opt_path_type_configs[i])
                    
#             else:
                
#                 _, _, phi_1, param_seg_2, phi_3 = CCC_path(ini_config, fin_config,\
#                                                            rL_opt, rR_opt, muL_opt, muR_opt,\
#                                                            opt_path_type_configs[i])
                    
#             # Obtaining the coordinates of points along the path
#             pts_path_x_coords, pts_path_y_coords = points_path(ini_config, fin_config,\
#                                                                rL_opt, rR_opt,\
#                                                                np.array([phi_1, param_seg_2, phi_3]),\
#                                                                opt_path_type_configs[i])
                
#             # Plotting the optimal path
#             fig_plane_path.scatter_2D(pts_path_x_coords, pts_path_y_coords,\
#                                       plot_colors[i], opt_path_configs[i], 3, plot_styles[i])
                
#         # Updating the layout and adding the figure to the html file
#         fig_plane_path.update_layout_2D('x (m)', [min(ini_config[0], fin_config[0]) - 3*max(rL, rR),\
#                                              max(ini_config[0], fin_config[0]) + 3*max(rL, rR)],\
#                                         'y (m)', [min(ini_config[1], fin_config[1]) - 3*max(rL, rR),\
#                                              max(ini_config[1], fin_config[1]) + 3*max(rL, rR)],\
#                                         'Optimal paths for given configurations')
#         # Writing onto the html file
#         fig_plane_path.writing_fig_to_html(filename, 'a')
                
    # return path_lengths_sym_unweighted, path_lengths_sym_weighted, path_costs_sym_weighted,\
    #     path_lengths_asym_unweighted, path_lengths_asym_weighted, path_costs_asym_weighted        