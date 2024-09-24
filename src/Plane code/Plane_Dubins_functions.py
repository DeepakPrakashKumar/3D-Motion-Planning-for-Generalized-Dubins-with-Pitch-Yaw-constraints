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
import sys

# Including the following command to ensure that python is able to find the relevant files afer changing directory
sys.path.insert(0, '')
# Obtaining the current directory
cwd = os.getcwd()
# Changing to one directory higher (since plotting class is located under Cylinder code)
os.chdir("..")
# Obtaining current common directory
common_dir = os.getcwd()

# Importing the plotting class functions
plotting_class_rel_path = '\Cylinder code'
os.chdir(common_dir + plotting_class_rel_path)
from plotting_class import plotting_functions

# Changing the directory back to the original path
os.chdir(cwd)

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

def points_path(ini_config, r, params_seg, path_type):
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
    r : Scalar
        Radius of the tight turn.
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
    
    # Checking if the length of the path_type string is the same as the number
    # of parameters provided in the params_seg array
    if len(path_type) != len(params_seg):
        
        raise Exception('The number of parameters provided corresponding to the path ' + \
                        'do not match with the number of segments in the path')
    
    # Obtaining the points along the path
    config_after_ith_segment = ini_config
    
    x_coords_path = np.array([])
    y_coords_path = np.array([])
    for i in range(len(path_type)):
        
        # Obtaining the points along the ith segment
        if path_type[i].lower() in ['l', 'r']:
        
            pts_ith_segment = Seg_pts(config_after_ith_segment, params_seg[i],
                                      r, path_type[i].lower())
                
        else:
            
            pts_ith_segment = Seg_pts(config_after_ith_segment, params_seg[i],\
                                      0, 's')
                
        # Appending the obtained points to the arrays
        x_coords_path = np.append(x_coords_path, pts_ith_segment[:, 0])
        y_coords_path = np.append(y_coords_path, pts_ith_segment[:, 1])
        
        # Updating the variable config_after_ith_segment
        config_after_ith_segment = pts_ith_segment[-1]
    
    return x_coords_path, y_coords_path

def CSC_path(ini_config, fin_config, r, path_type = 'lsl'):
    '''
    This function generates a CSC path connecting the initial and final configurations.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    r : Scalar
        Radius of the tight turn.
    path_type : String
        Contains the path type, and contains three segments made up of 'l' for left turn,
        'r' for right turn, and 's' for straight line segment.

    Returns
    -------
    path_length : Scalar
        Length of the path.
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
    
        lS2 = d**2 + 2*r**2 + 2*d*r*(math.sin(alpha_i) - math.sin(alpha_f))\
            - 2*(r**2)*math.cos(alpha_f - alpha_i)
        
    elif path_type.lower() == 'rsr':
        
        lS2 = d**2 + 2*r**2 + 2*d*r*(math.sin(alpha_f) - math.sin(alpha_i))\
            - 2*(r**2)*math.cos(alpha_f - alpha_i)
            
    elif path_type.lower() == 'lsr':
        
        lS2 = d**2 + 2*d*(r*math.sin(alpha_i) + r*math.sin(alpha_f))\
            + 2*r*r*(math.cos(alpha_f - alpha_i) - 1)
            
    elif path_type.lower() == 'rsl':
        
        lS2 = d**2 - 2*d*(r*math.sin(alpha_i) + r*math.sin(alpha_f)) \
            + 2*r*r*(math.cos(alpha_f - alpha_i) - 1)
            
    # Accounting for numerical inaccuracies
    if lS2 < 0 and lS2 >= -10**(-6):
        
        lS2 = 0
            
    # Checking if the path exists
    if lS2 < 0:
        
        print(path_type.upper() + ' path does not exist.')
        path_length = np.NaN
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
                path_length = r*phi_1
                
            elif path_type[0].lower() == 'r':
                
                phi_1 = np.mod((alpha_i - alpha_f), 2*math.pi)
                lS = 0
                phi_3 = 0
                path_length = r*phi_1
                
        # Checking if the length of the straight line segment equals zero for
        # LSR and RSL paths. If yes, the paths become a degenerate LR or RL path
        elif lS2 <= 10**(-6):
            
            print('Path is of type '+ path_type[0].upper() + path_type[2].upper() + '.')
            
            if path_type[0].lower() == 'l':
                
                # Angle of the first arc
                # Here, delta := atan2((rL + rR), ls) is set as pi/2.
                phi_1 = np.mod(math.pi/2 - alpha_i\
                               + math.atan2(-(r*math.cos(alpha_i) + r*math.cos(alpha_f)),\
                                            d + r*math.sin(alpha_i) + r*math.sin(alpha_f)),\
                               2*math.pi)
                # Angle of the final arc
                phi_3 = np.mod((alpha_i - alpha_f + phi_1), 2*math.pi)
                # Length of the straight line segment
                lS = 0
                
                # Path length
                path_length = r*phi_1 + lS + r*phi_3
                
            else:
                
                # Angle of the first arc
                # Here, delta := atan2((rL + rR), ls) is set as pi/2.
                phi_1 = np.mod(math.pi/2 + alpha_i\
                               - math.atan2(r*math.cos(alpha_i) + r*math.cos(alpha_f),\
                                            d - r*math.sin(alpha_i) - r*math.sin(alpha_f)),\
                               2*math.pi)
                # Angle of the final arc
                phi_3 = np.mod((alpha_f - alpha_i + phi_1), 2*math.pi)
                # Length of the straight line segment
                lS = 0
                
                # Path lengthcost
                path_length = r*phi_1 + lS + r*phi_3
                
        else:
            
            # Length of the straight line segment
            lS = math.sqrt(lS2)
            
            # Computing the angles for the first and the last arcs
            # LSL path
            if path_type.lower() == 'lsl':
                
                # Angle of the first arc
                phi_1 = np.mod(math.atan2(r*(math.cos(alpha_f) - math.cos(alpha_i)),\
                                          d - r*(math.sin(alpha_f) - math.sin(alpha_i)))\
                               - alpha_i, 2*math.pi)
                # Angle of the final arc
                phi_3 = np.mod(alpha_f - alpha_i - phi_1, 2*math.pi)
                
                # Path length
                path_length = r*(phi_1 + phi_3) + lS
                
            # RSR path
            elif path_type.lower() == 'rsr':
                
                # Angle of the first arc
                phi_1 = np.mod(-math.atan2(-r*(math.cos(alpha_f) - math.cos(alpha_i)),\
                                           d + r*(math.sin(alpha_f) - math.sin(alpha_i)))\
                               + alpha_i, 2*math.pi)
                # Angle of the final arc
                phi_3 = np.mod(alpha_i - alpha_f - phi_1, 2*math.pi)
                
                # Path length
                path_length = r*(phi_1 + phi_3) + lS
                
            # LSR path
            elif path_type.lower() == 'lsr':
                
                # Angle of the first arc
                phi_1 = np.mod(math.atan2((2*r), lS) - alpha_i\
                               + math.atan2(-(r*math.cos(alpha_i) + r*math.cos(alpha_f)),\
                                            d + r*math.sin(alpha_i) + r*math.sin(alpha_f)),\
                               2*math.pi)
                # Angle of the final arc
                phi_3 = np.mod((alpha_i - alpha_f + phi_1), 2*math.pi)
                
                # Path length
                path_length = r*phi_1 + lS + r*phi_3
                
            # RSL path
            elif path_type.lower() == 'rsl':
                
                # Angle of the first arc
                phi_1 = np.mod(math.atan2((2*r), lS) + alpha_i\
                               - math.atan2(r*math.cos(alpha_i) + r*math.cos(alpha_f),\
                                            d - r*math.sin(alpha_i) - r*math.sin(alpha_f)),\
                               2*math.pi)
                # Angle of the final arc
                phi_3 = np.mod((alpha_f - alpha_i + phi_1), 2*math.pi)
                
                # Path length
                path_length = r*phi_1 + lS + r*phi_3
        
    return path_length, phi_1, lS, phi_3

def CCC_path(ini_config, fin_config, r, path_type = 'lrl'):
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
    r : Scalar
        Radius of the tight turn.
    path_type : String
        Contains the path type, and contains three segments made up of 'l' for left turn,
        'r' for right turn, and 's' for straight line segment.

    Returns
    -------
    path_length : Scalar
        Length of the path.
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
        
        cos_phi_2 = 1 - (1/(2*(2*r)**2))*(d**2 + 2*r**2\
                                              - 2*d*r*(math.sin(alpha_f) - math.sin(alpha_i))\
                                              - 2*(r**2)*math.cos(alpha_f - alpha_i))
            
    elif path_type.lower() == 'rlr':
        
        cos_phi_2 = 1 - (1/(2*(2*r)**2))*(d**2 + 2*r**2\
                                              + 2*d*r*(math.sin(alpha_f) - math.sin(alpha_i))\
                                              - 2*(r**2)*math.cos(alpha_f - alpha_i))
            
    # Accounting for numerical inaccuracies
    if abs(cos_phi_2) > 1 and abs(cos_phi_2) <= 1 + 10**(-6):
        
        cos_phi_2 = np.sign(cos_phi_2)
            
    # Checking if the path exists
    if abs(cos_phi_2) > 1:
        
        print(path_type.upper() + ' path does not exist.')
        path_length = np.NaN
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
            
    else:
        
        # Choosing phi_2 such that it is always greater than pi. In the case that
        # cos_phi_2 = -1, phi_2 = pi.
        phi_2 = 2*math.pi - math.acos(cos_phi_2)
        
        # Computing the angles of the first and last segments
        # LRL path
        if path_type.lower() == 'lrl':
            
            # Angle of the first arc
            phi_1 = np.mod(math.atan2(r*(math.cos(alpha_f) - math.cos(alpha_i)),\
                                      d - r*(math.sin(alpha_f) - math.sin(alpha_i)))\
                           - alpha_i + phi_2/2, 2*math.pi)
            # Angle of the final arc
            phi_3 = np.mod((alpha_f - alpha_i - phi_1 + phi_2), 2*math.pi)
            
            # Path length
            path_length = r*(phi_1 + phi_2 + phi_3)
            
        elif path_type.lower() == 'rlr':
            
            # Angle of the first arc
            phi_1 = np.mod(-math.atan2(-r*(math.cos(alpha_f) - math.cos(alpha_i)),\
                                       d + r*(math.sin(alpha_f) - math.sin(alpha_i)))\
                           + alpha_i + phi_2/2, 2*math.pi)            
            # Angle of the final arc
            phi_3 = np.mod((alpha_i - alpha_f - phi_1 + phi_2), 2*math.pi)
            
            # Path length
            path_length = r*(phi_1 + phi_2 + phi_3)
            
        # Removing degenerate CCC paths
        if phi_1 <= 10**(-6) or phi_2 <= 10**(-6) or phi_3 <= 10**(-6):
            
            phi_1 = np.NaN
            phi_2 = np.NaN
            phi_3 = np.NaN
            path_length = np.NaN
        
    return path_length, phi_1, phi_2, phi_3

def dubins_paths(ini_config, fin_config, r, path_type = 'lsl'):
    '''
    This function generates the appropriate path required for the given configurations
    and parameters using the previously defined functions.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    r : Scalar
        Radius of the tight turn.
    path_type : String, optional
        Contains the path type to be generated. The default is 'lsl'.

    Returns
    -------
    path_length : Scalar
        Length of the path.
    params_path : Numpy array
        Contains the parameters of the path.

    '''
    
    # Listing the path types depending on the class of the paths it belongs to
    path_types_CSC = np.array(['lsl', 'rsr', 'lsr', 'rsl'])
    path_types_CCC = np.array(['lrl', 'rlr'])
    
    params_path = np.full(3, np.NaN)
    
    # Checking which type of the path has been passed and calling the appropriate function
    if path_type in path_types_CSC:
        
        path_length, params_path[0], params_path[1], params_path[2]\
            = CSC_path(ini_config, fin_config, r, path_type)
            
    elif path_type in path_types_CCC:
        
        path_length, params_path[0], params_path[1], params_path[2]\
            = CCC_path(ini_config, fin_config, r, path_type)
        
    else:
        
        raise Exception('Path type passed is invalid.')
    
    return path_length, params_path

def optimal_dubins_path(ini_config, fin_config, r, filename = 'plots_Dubins_paths.html'):
    '''
    This function generates the paths for 2D Dubins, and outputs the optimal path.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    r : Scalar
        Radius of the tight turn.
    filename : String, optional
        Name of the html file in which the plots are generated. The default is
        'plots_Dubins_paths.html'. If "False" is passed, then the plot is not generated.

    Returns
    -------
    path_lengths : 
    opt_path_type_configs : 
    pts_path_x_coords_opt : Numpy Nx1 array
        Contains the x-coordinates of points along the optimal path.
    pts_path_y_coords_opt : Numpy Nx1 array
        Contains the y-coordinates of points along the optimal path.

    '''
    
    path_types = np.array(['lsl', 'rsr', 'lsr', 'rsl', 'lrl', 'rlr'])
    path_lengths = np.empty(len(path_types))
    path_params = np.zeros((len(path_types), 3))
    
    if filename != False:
    
        # Creating an array for writing the lengths and costs of each path onto the html file.
        text_paths_file_asym_unweight = []
        text_paths_file_asym_unweight.append("---------Details of the configurations---------")
        text_paths_file_asym_unweight.append("The initial and final configurations are " + str(ini_config) +\
                                             ", " + str(fin_config))
        text_paths_file_asym_unweight.append("The parameter r is " + str(r))
        text_paths_file_asym_unweight.append("")
        text_paths_file_asym_unweight.append("---------Details of the paths for 2D Dubins---------")
    
    # Generating the paths for each type
    for i in range(len(path_types)):
                
        # Generating the paths
        path_lengths[i], path_params[i] = dubins_paths(ini_config, fin_config, r, path_types[i])
                
        # Adding to the string to be printed if the path exists
        if np.isnan(path_lengths[i]) == False and filename != False:
            
            temp = "Length of " + path_types[i].upper() + " path is " + str(path_lengths[i])
            text_paths_file_asym_unweight.append(temp)
            # Appending the parameters corresponding to the path also
            temp = "The parameters of the " + path_types[i].upper() + \
                " path are " + str(path_params[i])
            text_paths_file_asym_unweight.append(temp)
                
    # Finding the optimal path type for each combination
    opt_path_length = min(path_lengths)
    opt_path_type_configs = path_types[np.nanargmin(path_lengths)]
    
    # Obtaining points along the optimal paths for 2D Dubins
    path_params_opt = path_params[np.nanargmin(path_lengths)]
    pts_path_x_coords_opt, pts_path_y_coords_opt\
        = points_path(ini_config, r, path_params_opt, opt_path_type_configs)
    
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
        fig_plane.update_layout_2D('x (m)', [min(ini_config[0], fin_config[0]) - 3*r,\
                                              max(ini_config[0], fin_config[0]) + 3*r],\
                                    'y (m)', [min(ini_config[1], fin_config[1]) - 3*r,\
                                              max(ini_config[1], fin_config[1]) + 3*r],\
                                    'Initial and final configurations')
            
        # Writing onto the html file
        fig_plane.writing_fig_to_html(filename, 'w')
        
        # Adding the details of the paths to the html file
        with open(filename, 'a') as f:
            f.write("<br>")
            for i in range(len(text_paths_file_asym_unweight)):
                f.write(text_paths_file_asym_unweight[i] + ".<br />")
        
        # Creating the plots for the different path types
        for i in range(len(path_types)):
            
            # Making a copy of the created figure
            fig_plane_path = copy.deepcopy(fig_plane)             
                        
            # Plotting the path if the path exists
            if np.isnan(path_lengths[i]) == False:
            
                # Obtaining the coordinates of points along the path using the
                # points path function
                pts_path_x_coords, pts_path_y_coords = points_path(ini_config, r,\
                                                                   path_params[i],\
                                                                   path_types[i])
                    
                # Adding the path onto the figure
                fig_plane_path.scatter_2D(pts_path_x_coords, pts_path_y_coords,\
                                          'blue', 'Path', 3)
                    
                # Adding labels to the axis and title to the plot
                fig_plane_path.update_layout_2D('x (m)', [min(ini_config[0], fin_config[0]) - 3*r,\
                                                      max(ini_config[0], fin_config[0]) + 3*r],\
                                                'y (m)', [min(ini_config[1], fin_config[1]) - 3*r,\
                                                      max(ini_config[1], fin_config[1]) + 3*r],\
                                                path_types[i].upper() + ' path')
                # Writing onto the html file
                fig_plane_path.writing_fig_to_html(filename, 'a')
                    
        # Plotting the optimal path
        
        # Making a copy of the created figure
        fig_plane_path = copy.deepcopy(fig_plane)
        text_paths_file_optimal = []
        
        text_paths_file_optimal.append("The optimal path for 2D Dubins is of type "\
                                       + str(opt_path_type_configs.upper()) + \
                                       ', whose length is ' + str(opt_path_length) +\
                                       ', and whose parameters are '\
                                       + str(path_params_opt))
        # Plotting the optimal path
        fig_plane_path.scatter_2D(pts_path_x_coords_opt, pts_path_y_coords_opt,\
                                  'red', 'Optimal path', 3)
                
        # Updating the layout and adding the figure to the html file
        fig_plane_path.update_layout_2D('x (m)', [min(ini_config[0], fin_config[0]) - 3*r,\
                                                  max(ini_config[0], fin_config[0]) + 3*r],\
                                        'y (m)', [min(ini_config[1], fin_config[1]) - 3*r,\
                                                  max(ini_config[1], fin_config[1]) + 3*r],\
                                        'Optimal path for given configuration')
            
        with open(filename, 'a') as f:
            f.write("<br>")
            for i in range(len(text_paths_file_optimal)):
                f.write(text_paths_file_optimal[i] + ".<br />")
                
        # Writing onto the html file
        fig_plane_path.writing_fig_to_html(filename, 'a')
                    
    return path_lengths, opt_path_type_configs, pts_path_x_coords_opt, pts_path_y_coords_opt