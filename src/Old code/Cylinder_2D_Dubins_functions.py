# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 09:50:12 2021

@author: deepa
"""

import numpy as np
import math
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as py
import os
pio.renderers.default='browser'
import copy

# Loading the plotting class
from plotting_class import plotting_functions

# Loading the Dubins code
path_codes = 'D:\TAMU\Research\TwoD_Dubins_python_code'

os.chdir(path_codes)
from Dubins_functions import CCC_path, CSC_path
from Dubins_functions import CCC_path_efficient, CSC_path_efficient, points_path,\
    tight_circles, Seg_pts

def transformation_point_2D(ini_pos, ini_tang_vect, R, point_pos, point_tang_vect,\
                            # visualization = 1, 
                            zmax = 20):
    '''
    This function transforms a given point on a cylinder to a point on the 2D
    (unwrapping) plane. The unwrapping is done such that the origin of the plane
    coincides with the initial position.

    Parameters
    ----------
    ini_pos : Numpy 1x3 array
        Contains the coordinates of the initial position on the cylinder.
    ini_tang_vect : Numpy 1x3 array
        Contains the direction cosines of the tangent vector at the coordinates
        given by ini_pos. The direction cosines are given in the global frame.
    R : Scalar
        Radius of the cylinder.
    point_pos : Numpy 1x3 array
        Coordinates of a point on the cylinder that needs to be transformed to the plane.
    point_tang_vect : Numpy 1x3 array
        Contains the direction cosines of the tangent vector at the coordinates
        given by point_pos. The direction cosines are given in the global frame.
    visualization : Scalar
        visualization = 1 visualizes the two points with their headings on the cylinder
        and visualizes the location and heading of these points when unwrapped.
    zmax : Scalar
        Maximum value corresponding to the z-coordinate of points considered.

    Returns
    -------
    pos_plane_1, pos_plane_2 : Numpy 1x2 arrays
        Contains the coordinates of the provided point in the xz (unwrapping) plane.
        Two points are possible depending on whether unwrapping is done from theta in [0, 2pi)
        or theta in (-2pi, 0]. For the first type of unwrapping, coordinates is given by pos_plane_1
        and for the second type of unwrapping, coordinates is given by pos_plane_2.
    heading_ini_pos_plane, heading_point_pos_plane : Scalars
        Contains the heading at the point whose coordinates are given by ini_pos and
        point_pos, respectively.
    '''
    
    # Checking if the points lie on the cylinder or not
    tol = 10**(-3) # tolerance for checking if points lie on the cylinder or not
    if abs(ini_pos[0]**2 + ini_pos[1]**2 - R**2) > tol:
        
        raise Exception('Initial position does not lie on the cylinder.')
    if abs(point_pos[0]**2 + point_pos[1]**2 - R**2) > tol:
        
        raise Exception('Given position does not lie on the cylinder.')
    
    # Angle of the initial and given position corresponding to parametrization
    # in the global frame
    
    theta0 = math.atan2(ini_pos[1], ini_pos[0])
    theta = math.atan2(point_pos[1], point_pos[0])
    
    # Angle of point with respect to the initial point
    theta1 = theta - theta0
    # Angle taken in the counter-clockwise direction (rotation in the "left" sense) as positive
    if theta1 < 0:
        theta1 += 2*math.pi
        
    # Angle taken in the clockwise direction (rotation is the "right" sense) as positive
    # if theta1 < 0:
    #     theta2 = theta1
    # elif theta1 > 0:
    theta2 = theta1 - 2*math.pi
    
    # Coordinates of the point point_pos on the plane depending on how unwrapping is done
    pos_plane_1 = np.array([R*theta1, point_pos[2] - ini_pos[2]])
    pos_plane_2 = np.array([R*theta2, point_pos[2] - ini_pos[2]])
    
    # Heading at the initial position and the given position
    heading_ini_pos_plane = math.atan2(ini_tang_vect[2],\
                                       (-ini_tang_vect[0]*math.sin(theta0)\
                                        + ini_tang_vect[1]*math.cos(theta0)))
    heading_point_pos_plane = math.atan2(point_tang_vect[2],\
                                       (-point_tang_vect[0]*math.sin(theta)\
                                        + point_tang_vect[1]*math.cos(theta)))
        
    # if visualization == 1: # Plotting
        
    #     generate_visualize_path(ini_pos, ini_tang_vect, R, zmax, point_pos, point_tang_vect,\
    #                             pos_plane_1, pos_plane_2, heading_ini_pos_plane,\
    #                             heading_point_pos_plane)
        
    
    return pos_plane_1, pos_plane_2, heading_ini_pos_plane, heading_point_pos_plane

def homogeneous_transformation_matrix(ini_pos, R):
    '''
    This function returns the homogeneous transformation matrix relating the global
    frame to the body frame fixed at the initial position.

    Parameters
    ----------
    ini_pos : Numpy 1x3 array
        Contains the coordinates of the initial position on the cylinder.
    R : Scalar
        Radius of the cylinder.

    Returns
    -------
    H : Numpy 4x4 array
        Homogeneous transformation matrix relating the coordinates of a point in the
        global frame and the body frame fixed at the initial position.
    Hinv : Numpy 4x4 array
        Inverse of the H matrix.
    '''
    
    # Angle of the initial position corresponding to parametrization in the global frame
    
    theta0 = math.atan2(ini_pos[1], ini_pos[0])
    
    H = np.array([[-math.sin(theta0), -math.cos(theta0), 0, R*math.cos(theta0)],\
                  [math.cos(theta0), -math.sin(theta0), 0, R*math.sin(theta0)],\
                  [0, 0, 1, ini_pos[2]],\
                  [0, 0, 0, 1]])
    Hinv = np.array([[-math.sin(theta0), math.cos(theta0), 0, 0],\
                  [-math.cos(theta0), -math.sin(theta0), 0, R],\
                  [0, 0, 1, -ini_pos[2]],\
                  [0, 0, 0, 1]])
    
    return H, Hinv

def mapping_line_segment(pt_start_line, pt_end_line, R, ini_pos):
    '''
    This function maps a line segment given on a plane to the corresponding curve
    on a right circular cylinder of radius of R.

    Parameters
    ----------
    pt_start_line : Numpy 1x2 array
        Contains the coordinates corresponding to the start of the line segment on the plane.
    pt_end_line : Numpy 1x2 array
        Contains the coordinates corresponding to the end of the line segment on the plane.
    x_lower : Scalar
        Lower bound corresponding to the mapping from the cylinder to the plane.
    x_upper : Scalar
        Upper bound corresponding to the mapping from the cylinder to the plane.
    R : Scalar
        Radius of the cylinder.
    ini_pos : Numpy 1x3 array
        Contains the coordinates of the initial position on the cylinder.

    Returns
    -------
    points_curve_on_cylinder_global : Numpy array of size nx3
        Contains the points of the curve corresponding to the line segment on the cylinder in
        the global frame. For this purpose, the function homogeneous_transformation_matrix is
        used to convert the coordinates from the body frame to the global frame.
    points_curve_on_cylinder_body : Numpy array of size nx3
        Contains the points of the curve corresponding to the line segment on the cylinder in
        the body frame.
    '''
            
    # Equation of the line
    # Slope of the line
    alpha = (pt_end_line[1] - pt_start_line[1])/(pt_end_line[0] - pt_start_line[0])
    # Intersept of the line
    beta = pt_start_line[1] - alpha*pt_start_line[0]
    
    # Generating the points on the cylinder
    x_plane = np.linspace(pt_start_line[0], pt_end_line[0], 100)
    x = np.array([R*math.sin(i/R) for i in x_plane])
    y = np.array([R*(1 - math.cos(i/R)) for i in x_plane])
    z = np.array([alpha*i + beta for i in x_plane])
    
    points_curve_on_cylinder_global = np.zeros((np.size(x_plane), 3))
    points_curve_on_cylinder_body = np.zeros((np.size(x_plane), 3))
    
    for i in range(len(x_plane)):
        
        points_curve_on_cylinder_body[i, :] = np.array([x[i], y[i], z[i]])
        # Coordinates in the global frame
        H, Hinv = homogeneous_transformation_matrix(ini_pos, R)
        temp = np.matmul(H, np.array([x[i], y[i], z[i], 1]))
        points_curve_on_cylinder_global[i, :] = temp[0: -1]
    
    return points_curve_on_cylinder_global, points_curve_on_cylinder_body

def mapping_arc(pt_start_arc, pt_end_arc, center_circle_turn, rad_turn, sense_arc,\
                R, ini_pos, full_circle = 'n'):
    '''
    This function maps an arc of a circle of radius rad_turn given on a plane
    to the corresponding curve on a right circular cylinder of radius of R.

    Parameters
    ----------
    pt_start_arc : Numpy 1x2 array
        Contains the coordinates corresponding to the start of the arc on the plane.
    pt_end_arc : Numpy 1x2 array
        Contains the coordinates corresponding to the end of the arc on the plane.
    center_circle_turn : Numpy 1x2 array
        Contains the coordinates of the center of the circle on which the arc lies.
    rad_turn : Scalar
        Radius of the circle on which the arc lies on the plane.
    sense_arc : Character
        Contains the direction of motion from the start point of the arc to the
        end point of the arc described about an axis passing through the center 
        of the circle and out of the plane. 'l' indicates a left turn, i.e., a
        counter-clockwise turn, and 'r' indicates a right turn. Left turn is
        considered to be positive.
    R : Scalar
        Radius of the cylinder.
    ini_pos : Numpy 1x3 array
        Contains the coordinates of the initial position on the cylinder.
    full_circle : Character
        If 'y', a complete circle is mapped onto the cylinder. Default value is 'n'.

    Returns
    -------
    points_curve_on_cylinder_global : Numpy array of size nx3
        Contains the points of the curve corresponding to the arc on the cylinder in
        the global frame. For this purpose, the function homogeneous_transformation_matrix is
        used to convert the coordinates from the body frame to the global frame.
    points_curve_on_cylinder_body : Numpy array of size nx3
        Contains the points of the curve corresponding to the arc on the cylinder in
        the body frame.
    '''
    
    if sense_arc.lower() != 'l' and sense_arc.lower() != 'r':
        raise Exception("Incorrect sense of rotation.")
    
    # Angle of rotation from initial point to the final point on the circle
    # Angle of the initial point
    phi_initial = math.atan2(pt_start_arc[1] - center_circle_turn[1],\
                             pt_start_arc[0] - center_circle_turn[0])
        
    # Checking if points are such that a complete circle is required to be mapped
    if full_circle == 'y' and sense_arc.lower() == 'l':
        
        delta_phi = 2*math.pi
        
    elif full_circle == 'y' and sense_arc.lower() == 'r':
        
        delta_phi = - 2*math.pi
        
    # If not, finding the angle from the start to the end of the arc in the
    # direction mentioned
    else:
        
        phi_final = math.atan2(pt_end_arc[1] - center_circle_turn[1],\
                               pt_end_arc[0] - center_circle_turn[0])
        delta_phi = phi_final - phi_initial   
        
        if sense_arc.lower() == 'l' and delta_phi < -0.001: # to account for a numerical error,
        # a tolerance is given for checking
            
            delta_phi += 2*math.pi
            
        elif sense_arc.lower() == 'r' and delta_phi > 0.001: # to account for a numerical error,
        # a tolerance is given for checking
            
            delta_phi -= 2*math.pi
        
    # Generating the points on the cylinder
    phi_plane = np.linspace(phi_initial, phi_initial + delta_phi, 100)
    x = np.array([R*math.sin((center_circle_turn[0] + rad_turn*math.cos(i))/R)\
                  for i in phi_plane])
    y = np.array([R*(1 - math.cos((center_circle_turn[0] + rad_turn*math.cos(i))/R))\
                  for i in phi_plane])
    z = np.array([center_circle_turn[1] + rad_turn*math.sin(i) for i in phi_plane])
    
    points_curve_on_cylinder_global = np.zeros((np.size(phi_plane), 3))
    points_curve_on_cylinder_body = np.zeros((np.size(phi_plane), 3))
    
    for i in range(len(phi_plane)):
        
        points_curve_on_cylinder_body[i, :] = np.array([x[i], y[i], z[i]])
        # Coordinates in the global frame
        H, Hinv = homogeneous_transformation_matrix(ini_pos, R)
        temp = np.matmul(H, np.array([x[i], y[i], z[i], 1]))
        points_curve_on_cylinder_global[i, :] = temp[0: -1]

    return points_curve_on_cylinder_global, points_curve_on_cylinder_body

def generate_random_configs_cylinder(R, zmax):
    '''
    This function generates a random initial point on the cylinder, a random initial
    tangent vector at this point, and a random final point with a tangent vector
    on this cylinder. IT IS ASSUMED THAT THE ORIGIN OF THIS RIGHT CIRCULAR CYLINDER
    COINCIDES WITH THE GLOBAL ORIGIN AND THE AXIS OF THIS CYLINDER IS ALONG THE GLOBAL
    Z AXIS.

    Parameters
    ----------
    R : Scalar
        Radius of the cylinder.
    zmax : Scalar
        Maximum value corresponding to the z-coordinate for randomly generated points.

    Returns
    -------
    ini_pos : Numpy 1x3 array
        Coordinates corresponding to the initial point.
    ini_tang_vect : Numpy 1x3 array
        Direction cosines of the initial tangent vector.
    point_pos : Numpy 1x3 array
        Coordinates corresponding to a point that should be reached.
    point_tang_vect : Numpy 1x3 array
        Direction cosines of the random tangent vector at point_pos.
    '''
    
    # Generating initial and final positions
    length_scaling = zmax
    length_scaling_origin = length_scaling/2
    angle_scaling = 2*math.pi
    
    # Random initial and final angles for the initial and final position
    theta_ini = np.random.rand()*angle_scaling
    theta_fin = np.random.rand()*angle_scaling
    # Initial and final position
    ini_pos = np.array([R*math.cos(theta_ini), R*math.sin(theta_ini),\
                            np.random.rand()*length_scaling - length_scaling_origin])
    point_pos = np.array([R*math.cos(theta_fin), R*math.sin(theta_fin),\
                            np.random.rand()*length_scaling - length_scaling_origin])
        
    # Generating random tangent vectors and orthonormalizing using Gram-Schmidt
    # Finding the surface normal vectors corresponding to the initial and final positions
    uini = np.array([math.cos(theta_ini), math.sin(theta_ini), 0])
    ufin = np.array([math.cos(theta_fin), math.sin(theta_fin), 0])
    Tini_random = np.random.rand(3)
    Tfin_random = np.random.rand(3)
    # Orthonormalizing using Gram Schmidt
    wini = np.dot(Tini_random, uini)
    wfin = np.dot(Tfin_random, ufin)
    tol = 10**(-2) # tolerance for the dot product
    # Checking if Tini_random or Tfin_random are generated along tini or tfin, respectively.
    if abs(wini - np.linalg.norm(Tini_random)) < tol or\
        abs(wfin - np.linalg.norm(Tfin_random)) < tol:
        
        raise Exception('Regenerate the random vectors.')
        
    else:
        
        ini_tang_vect = (Tini_random - wini*uini)/np.linalg.norm(Tini_random - wini*uini)
        point_tang_vect = (Tfin_random - wfin*ufin)/np.linalg.norm(Tfin_random - wfin*ufin)
    
    return ini_pos, ini_tang_vect, point_pos, point_tang_vect

def Dubins_path_plane(ini_config, fin_config, rad_tight_turn, filename = False):
    '''
    This function generates the length of each three segment path connecting the
    initial and final configuration on a plane. The functions developed in the file
    Dubins_functions.py is used for this purpose. The details of the generated paths
    are written onto a html file if the argument filename is passed.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the x and y coordinate of the initial position and the heading angle.
    fin_config : Numpy 1x3 array
        Contains the x and y coordinate of the final position and the heading angle.
    rad_tight_turn : Scalar
        Radius of the tight circle turn.
    filename : String

    Returns
    -------
    path_types : Numpy 1x8 array
        Contains strings representing the path type.
    path_lengths : Numpy 1x8 array
        Contains the path length corresponding to each path type in the path_types variable.
        NaN is returned when a corresponding path ddoes not exist.

    '''
    
    path_types = np.array(['lsl', 'rsr', 'lsr', 'rsl', 'lrl', 'lrl', 'rlr', 'rlr'])
    path_lengths = np.empty(8)
    count_lrl = 1 # counter to ensure lrl path is run only once, as the CCC path generation
    # function generates both lrl paths
    count_rlr = 1 # counter to ensure rlr path is run only once
    
    for i in range(len(path_types)):
        
        # Checking if path is CSC type
        if path_types[i][1] == 's':
        
            # Obtaining the path length
            # path_lengths[i] = CSC_path(ini_config, fin_config, rad_tight_turn, path_types[i], 0)[0]
            path_lengths[i] = CSC_path_efficient(ini_config, fin_config, rad_tight_turn, path_types[i], 0)[0]
            
        # If not, path is of type CCC
        else:
            
            if (path_types[i][0] == 'l' and count_lrl == 1) or\
                (path_types[i][0] == 'r' and count_rlr == 1):
            
                # Obtaining the path length and adding to the array
                # path_lengths[i: i+2] = CCC_path(ini_config, fin_config, rad_tight_turn,\
                #                                 path_types[i], 0)[0]
                path_lengths[i: i+2] = CCC_path_efficient(ini_config, fin_config, rad_tight_turn,\
                                                path_types[i], 0)[0]
                
                # Incrementing the corresponding counter
                if path_types[i][0] == 'l':
                    
                    count_lrl += 1
                    
                else:
                    
                    count_rlr += 1
    
    if filename != False:
        
        # Printing details about paths that exist onto the html file
        text = []
        values = []
        # Writing number of possible paths
        text.append("Number of paths is ")
        values.append(np.count_nonzero(~np.isnan(path_lengths)))
        
        # Writing lengths of all the existing paths
        for i in range(len(path_types)):
            if np.isnan(path_lengths[i]) == False: # If the path exists
                temp = "Length of " + path_types[i].upper() + " path is "
                text.append(temp)
                values.append(path_lengths[i])
        
        # Writing onto the html file
        with open(filename, 'a') as f:
            f.write("<br>")
            for i in range(len(text)):
                f.write(text[i] + str(values[i]) + ".<br />")
    
    return path_types, path_lengths

def generate_visualize_path_simple(ini_pos, ini_tang_vect, R, fin_pos, fin_tang_vect,\
                                   zmax = 20, mode = 1, rad_tight_turn = 1, filename = 'test.html'):
    
    # Transforming the configurations to the plane    
    fin_pos_plane_1, fin_pos_plane_2, heading_ini_pos_plane, heading_fin_pos_plane = \
        transformation_point_2D(ini_pos, ini_tang_vect, R, fin_pos, fin_tang_vect, zmax)
        
    # Generating the initial and final configurations to pass to the Dubins functions
    ini_config = np.array([0, 0, heading_ini_pos_plane])
    fin_config_1 = np.array([fin_pos_plane_1[0], fin_pos_plane_1[1], heading_fin_pos_plane])
    fin_config_2 = np.array([fin_pos_plane_2[0], fin_pos_plane_2[1], heading_fin_pos_plane])
    
    path_types, path_lengths_img_1 = Dubins_path_plane(ini_config, fin_config_1,\
                                                       rad_tight_turn, False)
    _, path_lengths_img_2 = Dubins_path_plane(ini_config, fin_config_2, rad_tight_turn,\
                                              False)
        
    # Finding the optimal path
    min_path_length_img1 = min(path_lengths_img_1)
    min_path_length_img2 = min(path_lengths_img_2)
    
    # Finding p, q, and t for the optimal path
    if min_path_length_img1 <= min_path_length_img2:
        
        min_index = np.nanargmin(path_lengths_img_1)
        opt_fin_img = fin_config_1
        min_path_length = min_path_length_img1
        
    else:
        
        min_index = np.nanargmin(path_lengths_img_2)
        opt_fin_img = fin_config_2
        min_path_length = min_path_length_img2
        
    # Obtaining the parameters of the path corresponding to the optimal path
    if path_types[min_index][1] == 's':
        
        opt_path_length, t, p, q = CSC_path_efficient(ini_config, opt_fin_img, rad_tight_turn, path_types[min_index], 0)
        
    else:
        
        opt_path_length_arr, t_arr, p_arr, q_arr = CCC_path_efficient(ini_config, opt_fin_img, rad_tight_turn,\
                                                                      path_types[min_index], 0)
        # Since two paths exist for CCC path, we choose the least distance path out of the two
        opt_path_length = min(opt_path_length_arr)
        t = t_arr[np.nanargmin(opt_path_length_arr)]
        p = p_arr[np.nanargmin(opt_path_length_arr)]
        q = q_arr[np.nanargmin(opt_path_length_arr)]
        
    # Obtaining the points along the path using the points_path function
    pts_path_plane_x_coord, pts_path_plane_y_coord = points_path(ini_config, opt_fin_img, rad_tight_turn,\
                                                                 t, p, q, path_types[min_index])
        
    # Obtaining the points on the cylinder - using mapping_arc and mapping_seg functions
    # Obtaining the initial and final tight circles
    if path_types[min_index][0] == 'l':
        
        initial_circle = tight_circles(ini_config, rad_tight_turn, 0)[0, :]
        
    else:
        
        initial_circle = tight_circles(ini_config, rad_tight_turn, 0)[1, :]
        
    if path_types[min_index][2] == 'l':
        
        final_circle = tight_circles(opt_fin_img, rad_tight_turn, 0)[0, :]
        
    else:
        
        final_circle = tight_circles(opt_fin_img, rad_tight_turn, 0)[1, :]
        
    # Obtaining the tangent points (for arcs) and end of line segment (for straight line segment)
    # tang_pt_1 = Seg_pts(ini_config, ini_config, opt_fin_img, t, rad_tight_turn, path_types[min_index][0])[-1, :]
    # tang_pt_2 = Seg_pts(tang_pt_1, ini_config, opt_fin_img, p, rad_tight_turn, path_types[min_index][1])[-1, :]
    tang_pt_1 = Seg_pts(ini_config, t, rad_tight_turn, path_types[min_index][0])[-1, :]
    tang_pt_2 = Seg_pts(tang_pt_1, p, rad_tight_turn, path_types[min_index][1])[-1, :]
    
    points_seg_global_1,_ = mapping_arc(np.array([0, 0]), tang_pt_1, initial_circle, rad_tight_turn,\
                                            path_types[min_index][0], R, ini_pos)
    
    if path_types[min_index][1] == 's':
        
        points_seg_global_2,_ = mapping_line_segment(tang_pt_1, tang_pt_2, R, ini_pos)
        
    else:
    
        if path_types[min_index][1] == 'l':
            
            middle_circle = tight_circles(tang_pt_1, rad_tight_turn, 0)[0, :]
            
        elif path_types[min_index][1] == 'r':
            
            middle_circle = tight_circles(tang_pt_1, rad_tight_turn, 0)[1, :]
            
        points_seg_global_2,_ = mapping_arc(tang_pt_1, tang_pt_2, middle_circle, rad_tight_turn,\
                                            path_types[min_index][1], R, ini_pos)
        
    points_seg_global_3,_ = mapping_arc(tang_pt_2, opt_fin_img, final_circle, rad_tight_turn,\
                                            path_types[min_index][2], R, ini_pos)
        
    points_path_global = np.concatenate((np.concatenate((points_seg_global_1, points_seg_global_2)),\
                                    points_seg_global_3))
        
    if mode == 2:    
    
        # Creating a 3D plot. fig_cylinder is declared as an instance of the class
        # plotting_functions.
        fig_cylinder = plotting_functions()
        
        # Plotting the cylinder
        theta = np.linspace(0, 2*math.pi, 50)
        z = np.linspace(-zmax-5, zmax+5, 50)
        theta_grid, z_grid = np.meshgrid(theta, z, indexing = 'ij')
        x_grid = R*np.cos(theta_grid)
        y_grid = R*np.sin(theta_grid)
        
        fig_cylinder.surface_3D(x_grid, y_grid, z_grid, 'grey', False, 0.5)
        # Plotting the initial and final points
        fig_cylinder.points_3D([ini_pos[0]], [ini_pos[1]], [ini_pos[2]], 'red', 'Initial point',\
                               'circle')
        fig_cylinder.points_3D([fin_pos[0]], [fin_pos[1]], [fin_pos[2]], 'black', 'Final point',\
                               'diamond')
        # Adding initial and final tangent vectors
        fig_cylinder.arrows_3D([ini_pos[0]], [ini_pos[1]], [ini_pos[2]], [ini_tang_vect[0]],\
                              [ini_tang_vect[1]], [ini_tang_vect[2]], 'orange', 'oranges',\
                              'Initial tangent vector', 5, 5, 4, 'n')
        fig_cylinder.arrows_3D([fin_pos[0]], [fin_pos[1]], [fin_pos[2]], [fin_tang_vect[0]],\
                              [fin_tang_vect[1]], [fin_tang_vect[2]], 'green', 'greens',\
                              'Final tangent vector', 5, 5, 4, 'n')
        # Adding the axes for the body frame
        ini_pos_parametrization_angle = math.atan2(ini_pos[1], ini_pos[0])
        fig_cylinder.arrows_3D([ini_pos[0]], [ini_pos[1]], [ini_pos[2]],\
                               [-math.sin(ini_pos_parametrization_angle)],\
                               [math.cos(ini_pos_parametrization_angle)], [0],\
                               'brown', 'Brwnyl', 'Body frame', 5, 5, 4, 'y', 'x')
        fig_cylinder.arrows_3D([ini_pos[0]], [ini_pos[1]], [ini_pos[2]],\
                               [-math.cos(ini_pos_parametrization_angle)],\
                               [-math.sin(ini_pos_parametrization_angle)], [0],\
                               'brown', 'Brwnyl', False, 5, 5, 4, 'y', 'y')
        fig_cylinder.arrows_3D([ini_pos[0]], [ini_pos[1]], [ini_pos[2]], [0], [0],\
                               [1], 'brown', 'Brwnyl', False, 5, 5, 4, 'y', 'z')
            
        # Adding labels to the axis and title to the plot
        fig_cylinder.update_layout_3D('X (m)', 'Y (m)', 'Z (m)',\
                                      'Initial and final configurations on a cylinder')
        # Writing onto the html file
        # fig_cylinder.writing_fig_to_html(filename, 'w')
        fig_cylinder.writing_fig_to_html(filename, 'a')
        
        # Plotting the initial and final configuration on the plane after unwrapping the cylinder
        # Creating a 2D plot. fig_plane is declared as an instance of the class plotting_functions.
        fig_plane = plotting_functions()
        
        # Plotting the initial position and the two images of the final configuration on the plane
        fig_plane.points_2D([0], [0], 'red', 'Initial point', 'circle')
        fig_plane.points_2D([fin_pos_plane_1[0]], [fin_pos_plane_1[1]], 'black',\
                            'Final point - first image', 'diamond')
        fig_plane.points_2D([fin_pos_plane_2[0]], [fin_pos_plane_2[1]], 'brown',\
                            'Final point - second image', 'cross')
            
        # Adding initial and final headings
        fig_plane.arrows_2D([0], [0], [math.cos(heading_ini_pos_plane)],\
                            [math.sin(heading_ini_pos_plane)], 'orange', 'Initial heading', 2)
        fig_plane.arrows_2D([fin_pos_plane_1[0]], [fin_pos_plane_1[1]], [math.cos(heading_fin_pos_plane)],\
                            [math.sin(heading_fin_pos_plane)], 'green', 'Final heading', 2)
        fig_plane.arrows_2D([fin_pos_plane_2[0]], [fin_pos_plane_2[1]], [math.cos(heading_fin_pos_plane)],\
                            [math.sin(heading_fin_pos_plane)], 'green', False, 2)
            
        # Adding labels to the axis and title to the plot
        fig_plane.update_layout_2D('x (m)', [-2*(math.pi*R + rad_tight_turn), 2*(math.pi*R + rad_tight_turn)],\
                                   'z (m)', [-zmax-5, zmax+5], 'Initial and final configurations on a plane')
        # Writing onto the html file
        fig_plane.writing_fig_to_html(filename, 'a')
        
        # Obtaining the lengths of all paths to image 1 and image 2 of the final configuration
        # from the initial configuration and writing on the html file (done in the
        # Dubins_path_plane function)
        
        with open(filename, 'a') as f:
            f.write("<br\><br\><br\>")
        with open(filename, 'a') as f:
            f.write("-----------Paths to first image of the final configuration-----------")
        Dubins_path_plane(ini_config, fin_config_1, rad_tight_turn, filename)
        with open(filename, 'a') as f:
            f.write("-----------Paths to second image of the final configuration-----------")
        Dubins_path_plane(ini_config, fin_config_2, rad_tight_turn, filename)
        with open(filename, 'a') as f:
            f.write("Optimal path is of type " + path_types[min_index].upper() + " and of length "\
                    + str(opt_path_length) + ".<br />")
        
        # Drawing the optimal path on the plane
        fig_plane.scatter_2D(pts_path_plane_x_coord, pts_path_plane_y_coord, 'blue', 'Optimal path')
        # Adding labels to the axis and title to the plot
        fig_plane.update_layout_2D('x (m)', [-2*(math.pi*R + rad_tight_turn), 2*(math.pi*R + rad_tight_turn)],\
                                   'z (m)', [-zmax-5, zmax+5], 'Optimal path on the plane')
        # Writing onto the html file
        fig_plane.writing_fig_to_html(filename, 'a')
        
        # Drawing the optimal path on the cylinder
        fig_cylinder.scatter_3D(points_path_global[:, 0], points_path_global[:, 1],\
                                points_path_global[:, 2], 'blue', 'Optimal path')
        # Adding labels to the axis and title to the plot
        fig_cylinder.update_layout_3D('X (m)', 'Y (m)', 'Z (m)',\
                                      'Optimal path on the cylinder')
        # Writing onto the html file
        fig_cylinder.writing_fig_to_html(filename, 'a')
        
    return min_path_length, path_types[min_index], points_path_global

def generate_visualize_path(ini_pos, ini_tang_vect, R, point_pos, point_tang_vect,\
                            # pos_plane_1, pos_plane_2, heading_initial, heading_final,\
                            zmax = 20, mode = 1, rad_tight_turn = 1, filename = 'temp.html'):
    '''
    This function generates a path between two points on a right circular cylinder
    using the 2D Dubins result and visualizes the path.

    Parameters
    ----------
    ini_pos : TYPE
        DESCRIPTION.
    ini_tang_vect : TYPE
        DESCRIPTION.
    R : TYPE
        DESCRIPTION.
    point_pos : TYPE
        DESCRIPTION.
    point_tang_vect : TYPE
        DESCRIPTION.
    rad_tight_turn : TYPE
        DESCRIPTION.
    visualization : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    '''
    
    # Transforming the configurations to the plane    
    pos_plane_1, pos_plane_2, heading_ini_pos_plane, heading_point_pos_plane = \
        transformation_point_2D(ini_pos, ini_tang_vect, R, point_pos, point_tang_vect, zmax)
        
    if mode == 1 or mode == 2:
        
        fig_cylinder = go.Figure()
        
        # Plotting the cylinder
        theta = np.linspace(0, 2*math.pi, 50)
        z = np.linspace(-zmax-5, zmax+5, 50)
        theta_grid, z_grid = np.meshgrid(theta, z, indexing = 'ij')
        x_grid = R*np.cos(theta_grid)
        y_grid = R*np.sin(theta_grid)
        
        fig_cylinder.add_trace(go.Surface
                     (x = x_grid,
                      y = y_grid,
                      z = z_grid,
                      showscale = False,
                      colorscale = [[0, "grey"], [1, "grey"]],
                      opacity = 0.5
                         ))
        
        # Showing the initial and final points on the cylinder
        fig_cylinder.add_trace(go.Scatter3d
                     (x = [ini_pos[0]],
                      y = [ini_pos[1]],
                      z = [ini_pos[2]],
                      marker=dict(
                                size=8,
                                colorscale='Viridis',
                                symbol = 'circle',
                                color = 'red'),
                      name = 'Initial point'
                         ))
        fig_cylinder.add_trace(go.Scatter3d
                     (x = [point_pos[0]],
                      y = [point_pos[1]],
                      z = [point_pos[2]],
                      marker=dict(
                                size=8,
                                colorscale='Viridis',
                                symbol = 'diamond',
                                color = 'black'),
                      name = 'Final point'
                         ))
        
        # Showing initial and final tangent
        length_vec = 5
        arrowtipsize = 3
        fig_cylinder.add_trace(go.Scatter3d
                     (x = [ini_pos[0], ini_pos[0] + ini_tang_vect[0]*length_vec],
                      y = [ini_pos[1], ini_pos[1] + ini_tang_vect[1]*length_vec],
                      z = [ini_pos[2], ini_pos[2] + ini_tang_vect[2]*length_vec],
                      mode = "lines",
                      line = dict(color = 'orange', width = 5, colorscale = 'Viridis'),
                      name = 'Initial tangent vector'
                         ))
        fig_cylinder.add_trace(go.Scatter3d
                     (x = [point_pos[0], point_pos[0] + point_tang_vect[0]*length_vec],
                      y = [point_pos[1], point_pos[1] + point_tang_vect[1]*length_vec],
                      z = [point_pos[2], point_pos[2] + point_tang_vect[2]*length_vec],
                      mode = "lines",
                      line = dict(color = 'green', width = 5, colorscale = 'Viridis'),
                      name = 'Final tangent vector'
                         ))
        fig_cylinder.add_trace(go.Cone
                     (x = [ini_pos[0] + ini_tang_vect[0]*length_vec],
                      y = [ini_pos[1] + ini_tang_vect[1]*length_vec],
                      z = [ini_pos[2] + ini_tang_vect[2]*length_vec],
                      u = [arrowtipsize*(ini_tang_vect[0])],
                      v = [arrowtipsize*(ini_tang_vect[1])],
                      w = [arrowtipsize*(ini_tang_vect[2])],
                      colorscale = 'oranges',
                      showscale = False,
                      showlegend = False
                         ))
        fig_cylinder.add_trace(go.Cone
                     (x = [point_pos[0] + point_tang_vect[0]*length_vec],
                      y = [point_pos[1] + point_tang_vect[1]*length_vec],
                      z = [point_pos[2] + point_tang_vect[2]*length_vec],
                      u = [arrowtipsize*(point_tang_vect[0])],
                      v = [arrowtipsize*(point_tang_vect[1])],
                      w = [arrowtipsize*(point_tang_vect[2])],
                      colorscale = 'greens',
                      showscale = False,
                      showlegend = False
                         ))
        
        # Showing the body frame
        # x-axis
        fig_cylinder.add_trace(go.Scatter3d
                     (x = [ini_pos[0], ini_pos[0] - math.sin(math.atan2(ini_pos[1], ini_pos[0]))*length_vec],
                      y = [ini_pos[1], ini_pos[1] + math.cos(math.atan2(ini_pos[1], ini_pos[0]))*length_vec],
                      z = [ini_pos[2], ini_pos[2]],
                      mode = "lines+text",
                      line = dict(color = 'brown', width = 5, colorscale = 'Viridis'),
                      text = ['', 'x'],
                      textposition = 'top right',
                      name = 'Body frame',
                      textfont_size = 18
                         ))
        fig_cylinder.add_trace(go.Cone
                     (x = [ini_pos[0] - math.sin(math.atan2(ini_pos[1], ini_pos[0]))*length_vec],
                      y = [ini_pos[1] + math.cos(math.atan2(ini_pos[1], ini_pos[0]))*length_vec],
                      z = [ini_pos[2]],
                      u = [arrowtipsize*(- math.sin(math.atan2(ini_pos[1], ini_pos[0])))],
                      v = [arrowtipsize*(math.cos(math.atan2(ini_pos[1], ini_pos[0])))],
                      w = [arrowtipsize*(0)],
                      colorscale = 'Brwnyl',
                      showscale = False,
                      showlegend = False
                         ))
        # y-axis
        fig_cylinder.add_trace(go.Scatter3d
                     (x = [ini_pos[0], ini_pos[0] - math.cos(math.atan2(ini_pos[1], ini_pos[0]))*length_vec],
                      y = [ini_pos[1], ini_pos[1] - math.sin(math.atan2(ini_pos[1], ini_pos[0]))*length_vec],
                      z = [ini_pos[2], ini_pos[2]],
                      mode = "lines+text",
                      line = dict(color = 'brown', width = 5, colorscale = 'Viridis'),
                      text = ['', 'y'],
                      textposition = 'top right',
                      showlegend = False,
                      textfont_size = 18
                         ))
        fig_cylinder.add_trace(go.Cone
                     (x = [ini_pos[0] - math.cos(math.atan2(ini_pos[1], ini_pos[0]))*length_vec],
                      y = [ini_pos[1] - math.sin(math.atan2(ini_pos[1], ini_pos[0]))*length_vec],
                      z = [ini_pos[2]],
                      u = [arrowtipsize*(- math.cos(math.atan2(ini_pos[1], ini_pos[0])))],
                      v = [arrowtipsize*(- math.sin(math.atan2(ini_pos[1], ini_pos[0])))],
                      w = [arrowtipsize*(0)],
                      colorscale = 'Brwnyl',
                      showscale = False,
                      showlegend = False
                         ))
        # z-axis
        fig_cylinder.add_trace(go.Scatter3d
                     (x = [ini_pos[0], ini_pos[0]],
                      y = [ini_pos[1], ini_pos[1]],
                      z = [ini_pos[2], ini_pos[2] + length_vec],
                      mode = "lines+text",
                      line = dict(color = 'brown', width = 5, colorscale = 'Viridis'),
                      text = ['', 'z'],
                      textposition = 'top right',
                      showlegend = False,
                      textfont_size = 18
                         ))
        fig_cylinder.add_trace(go.Cone
                     (x = [ini_pos[0]],
                      y = [ini_pos[1]],
                      z = [ini_pos[2] + length_vec],
                      u = [arrowtipsize*(0)],
                      v = [arrowtipsize*(0)],
                      w = [arrowtipsize*(1)],
                      colorscale = 'Brwnyl',
                      showscale = False,
                      showlegend = False
                         ))
        
        fig_cylinder.update_layout(
            # width = 800,
            # height = 700,
            # autosize = False,
            scene = dict(
                camera = dict(
                    up = dict(
                        x = 0,
                        y = 0,
                        z = 1
                    ),
                    eye=dict(
                        # x = 0,
                        # y = 1.0707,
                        # z = 1,
                        x = 1.2,
                        y = 1.0707,
                        z = 1
                    )
                ),
                aspectratio = dict(x = 0.75, y = 0.75, z = 0.5),
                aspectmode = 'manual',
                xaxis_title = 'X (m)',
                yaxis_title = 'Y (m)',
                zaxis_title = 'Z (m)'
                # ,
                # uniformtext_minsize = 8
            ),
            title_text = 'Initial and final configurations on a cylinder'
        )
        
        with open(filename, 'w') as f:
            f.write(fig_cylinder.to_html(full_html=False, include_plotlyjs='cdn'))
            
        # Plotting the plane
        
        fig_plane = go.Figure()
        
        # Plotting the initial and final points (both images)
        fig_plane.add_trace(go.Scatter
                      (x = [0],
                       y = [0],
                       marker=dict(
                                size=8,
                                colorscale='Viridis',
                                symbol = 'circle',
                                color = 'red'),
                      name = 'Initial point'))
        fig_plane.add_trace(go.Scatter
                      (x = [pos_plane_1[0]],
                       y = [pos_plane_1[1]],
                       marker=dict(
                                size=8,
                                colorscale='Viridis',
                                symbol = 'diamond',
                                color = 'black'),
                      name = 'Final point - first image'))
        fig_plane.add_trace(go.Scatter
                      (x = [pos_plane_2[0]],
                       y = [pos_plane_2[1]],
                       marker=dict(
                                size=8,
                                colorscale='Viridis',
                                symbol = 'cross',
                                color = 'grey'),
                      name = 'Final point - second image'))
        # Plotting the headings
        length_vec = 2
        triangle_size = 2
        fig_plane.add_trace(go.Scatter
                      (x = [0, math.cos(heading_ini_pos_plane)*length_vec],
                       y = [0, math.sin(heading_ini_pos_plane)*length_vec],
                       mode = "lines",
                       line = dict(color = 'orange', width = 3),
                       name = 'Initial heading'
                       ))
        # fig.add_trace(go.Scatter
        #               (x = [math.cos(heading_ini_pos_plane)*length_vec\
        #                     + math.cos(heading_ini_pos_plane)*triangle_size,\
        #                     ]
        #                   ))
        fig_plane.add_trace(go.Scatter
                      (x = [pos_plane_1[0], pos_plane_1[0] + math.cos(heading_point_pos_plane)*length_vec],
                        y = [pos_plane_1[1], pos_plane_1[1] + math.sin(heading_point_pos_plane)*length_vec],
                        mode = "lines",
                        line = dict(color = 'green', width = 3),
                        name = 'Final heading'
                        ))
        fig_plane.add_trace(go.Scatter
                      (x = [pos_plane_2[0], pos_plane_2[0] + math.cos(heading_point_pos_plane)*length_vec],
                        y = [pos_plane_2[1], pos_plane_2[1] + math.sin(heading_point_pos_plane)*length_vec],
                        mode = "lines",
                        line = dict(color = 'green', width = 3),
                        showlegend = False
                        ))
        
        fig_plane.update_layout(xaxis_title = 'x (m)',
                          yaxis_title = 'z (m)',
                          title_text = 'Initial and final configurations on a plane',
                          xaxis_range = [-2*math.pi*R, 2*math.pi*R])
        
        with open(filename, 'a') as f:
            f.write(fig_plane.to_html(full_html=False, include_plotlyjs='cdn'))
            
    if mode == 2:
        
        # Generating the initial and final configurations to pass to the Dubins functions
        ini_config = np.array([0, 0, heading_ini_pos_plane])
        fin_config_1 = np.array([pos_plane_1[0], pos_plane_1[1], heading_point_pos_plane])
        fin_config_2 = np.array([pos_plane_2[0], pos_plane_2[1], heading_point_pos_plane])
        
        # Obtaining the paths on the plane to the first and second image of the final configuration
        path_types_CSC = np.array(['lsl', 'rsl', 'lsr', 'rsr'])
        path_lengths_CSC_img_1 = []
        path_lengths_CSC_img_2 = []
        
        for i in range(len(path_types_CSC)):
            
            temp = CSC_path(ini_config, fin_config_1, rad_tight_turn, path_types_CSC[i], 1)[0]
            path_lengths_CSC_img_1.append(temp)
            temp = CSC_path(ini_config, fin_config_2, rad_tight_turn, path_types_CSC[i], 1)[0]
            path_lengths_CSC_img_2.append(temp)
            
        path_lengths_CSC_img_1 = np.array(path_lengths_CSC_img_1)
        path_lengths_CSC_img_2 = np.array(path_lengths_CSC_img_2)
        
        path_types_CCC = np.array(['lrl', 'rlr'])
        path_lengths_CCC_img_1 = []
        path_lengths_CCC_img_2 = []
        
        for i in range(len(path_types_CCC)):
            
            temp = CCC_path(ini_config, fin_config_1, rad_tight_turn, path_types_CCC[i], 1)[0]
            path_lengths_CCC_img_1.append(temp[0])
            path_lengths_CCC_img_1.append(temp[1])
            temp = CCC_path(ini_config, fin_config_2, rad_tight_turn, path_types_CCC[i], 1)[0]
            path_lengths_CCC_img_2.append(temp[0])
            path_lengths_CCC_img_2.append(temp[1])
        
        path_lengths_CCC_img_1 = np.array(path_lengths_CCC_img_1)
        path_lengths_CCC_img_2 = np.array(path_lengths_CCC_img_2)
        
        # Concatenating the path lengths into a single array
        path_lengths_img_1 = np.concatenate((path_lengths_CSC_img_1, path_lengths_CCC_img_1))
        path_lengths_img_2 = np.concatenate((path_lengths_CSC_img_2, path_lengths_CCC_img_2))
        path_types = np.array([path_types_CSC[0], path_types_CSC[1], path_types_CSC[2], path_types_CSC[3],\
                      path_types_CCC[0], path_types_CCC[0], path_types_CCC[1], path_types_CCC[1]])
            
        # Writing on the html file
        text = []
        values = []
        # Writing number of possible paths
        text.append("----------------Paths to images of final configuration-----------")
        values.append("None")
        text.append("Number of paths from initial configuration to the first image of the final configuration is ")
        values.append(np.count_nonzero(~np.isnan(path_lengths_img_1)))        
        text.append("Number of paths from initial configuration to the second image of the final configuration is ")
        values.append(np.count_nonzero(~np.isnan(path_lengths_img_2)))
        
        for i in range(len(path_types)):
            if np.isnan(path_lengths_img_1[i]) == False:
                temp= "Length of " + path_types[i].upper() + " path to the first image of the final configuration is "
                text.append(temp)
                values.append(path_lengths_img_1[i])
            if np.isnan(path_lengths_img_2[i]) == False:
                temp = "Length of " + path_types[i].upper() + " path to the second image of the final configuration is "
                text.append(temp)
                values.append(path_lengths_img_2[i])
        
        with open(filename, 'a') as f:
            f.write("<br>")
            for i in range(len(text)):
                if values[i] == "None":
                    f.write(text[i] + ".<br />")
                else:
                    f.write(text[i] + str(values[i]) + ".<br />")
        
        # Plotting the plane with the path for all possible paths
        
        # Counters to decide whether to plot first CCC path or second CCC path
        count_lrl = 1 # for lrl path types
        count_rlr = 1 # for rlr path types
        
        for i in range(len(path_types)):
            
            # Plotting when the path corresponding to this path type exists between
            # the initial configuration and atleast one of the images of the final configuration
            if np.isnan(path_lengths_img_1[i]) == False or np.isnan(path_lengths_img_2[i]) == False:
        
                # To plot the path on the plane and the cylinder. For this purpose,
                # the already made plots for the plane and the cylinder are modified.
                fig_plane_path = copy.deepcopy(fig_plane)
                fig_cylinder_path = copy.deepcopy(fig_cylinder)                
                
                # Plotting the path
                # Getting the parameters for the path if the path exists
                # Checking if the middle segment is a straight line or arc
                if path_types[i][1] == 's':
                    
                    # Checking if the path exists to first image of the final configuration
                    if np.isnan(path_lengths_img_1[i]) == False:
                    
                        _, initial_circle, final_circle_1, tang_pts_1, angle_1_1,\
                            angle_2_1 = CSC_path(ini_config, fin_config_1,\
                                               rad_tight_turn, path_types[i], 0)
                    
                    # Checking if the path exists to second image of the final configuration
                    if np.isnan(path_lengths_img_2[i]) == False:
                    
                        _, initial_circle, final_circle_2, tang_pts_2, angle_1_2,\
                            angle_2_2 = CSC_path(ini_config, fin_config_2,\
                                                 rad_tight_turn, path_types[i], 0)
                            
                else:
                    
                    # Checking if the path exists to first image of the final configuration
                    # Nomenclature : final subscript denotes if parameters of path to first
                    # image or second image is constructed 
                    if np.isnan(path_lengths_img_1[i]) == False:
                        
                        _, initial_circle, final_circle_1, middle_circle_1_1,\
                            tang_pt_11_1, tang_pt_31_1, angle_11_1, angle_21_1, angle_31_1,\
                            middle_circle_2_1, tang_pt_12_1, tang_pt_32_1, angle_12_1,\
                            angle_22_1, angle_32_1 = CCC_path(ini_config, fin_config_1,\
                                                            rad_tight_turn, path_types[i], 0)
                                
                    # Checking if the path exists to second image of the final configuration
                    if np.isnan(path_lengths_img_2[i]) == False:
                        
                        _, initial_circle, final_circle_2, middle_circle_1_2,\
                            tang_pt_11_2, tang_pt_31_2, angle_11_2, angle_21_2, angle_31_2,\
                            middle_circle_2_2, tang_pt_12_2, tang_pt_32_2, angle_12_2,\
                            angle_22_2, angle_32_2 = CCC_path(ini_config, fin_config_2,\
                                                              rad_tight_turn, path_types[i], 0)
                
                # Plotting the initial and final tight circles
                fig_plane_path.add_trace(go.Scatter
                                         (x = np.array([initial_circle[0] + rad_tight_turn*math.cos(i)\
                                                        for i in np.linspace(0, 2*math.pi, 50)]),
                                          y = np.array([initial_circle[1] + rad_tight_turn*math.sin(i)\
                                                        for i in np.linspace(0, 2*math.pi, 50)]),
                                          mode = 'lines',
                                          line = dict(color = 'yellow', width = 3),
                                          name = 'Initial tight circle'))
                    
                fig_plane_path.add_trace(go.Scatter
                                         (x = np.array([final_circle_1[0] + rad_tight_turn*math.cos(i)\
                                                        for i in np.linspace(0, 2*math.pi, 50)]),
                                          y = np.array([final_circle_1[1] + rad_tight_turn*math.sin(i)\
                                                        for i in np.linspace(0, 2*math.pi, 50)]),
                                          mode = 'lines',
                                          line = dict(color = 'deeppink', width = 3),
                                          name = 'Final tight circle'))
                fig_plane_path.add_trace(go.Scatter
                                         (x = np.array([final_circle_2[0] + rad_tight_turn*math.cos(i)\
                                                        for i in np.linspace(0, 2*math.pi, 50)]),
                                          y = np.array([final_circle_2[1] + rad_tight_turn*math.sin(i)\
                                                        for i in np.linspace(0, 2*math.pi, 50)]),
                                          mode = 'lines',
                                          line = dict(color = 'deeppink', width = 3),
                                          showlegend = False))
                    
                # Plotting the paths
                # If path exists to the first image from the initial configuration
                if np.isnan(path_lengths_img_1[i]) == False:
                
                    # Initial configuration's angle wrt initial tight circle
                    phi0 = math.atan2(ini_config[1] - initial_circle[1],\
                                      ini_config[0] - initial_circle[0])
                    # Final configuration's angle wrt final tight circle
                    phi2 = math.atan2(fin_config_1[1] - final_circle_1[1],\
                                      fin_config_1[0] - final_circle_1[0])    
                
                    # Plotting for CSC path
                    if path_types[i][1].lower() == 's':
                        
                        # Creating the ranges for the angles
                        phi_circle_1 = np.linspace(phi0, phi0 + angle_1_1, 100)
                        phi_circle_2 = np.linspace(phi2 - angle_2_1, phi2, 100)
                
                        # Plotting the first arc
                        fig_plane_path.add_trace(go.Scatter
                                      (x = np.array([initial_circle[0] +\
                                                     rad_tight_turn*math.cos(i)\
                                                     for i in phi_circle_1]),
                                       y = np.array([initial_circle[1] +\
                                                     rad_tight_turn*math.sin(i)\
                                                     for i in phi_circle_1]),
                                       mode = 'lines',
                                       line = dict(color = 'blue', width = 3),
                                       name = 'Path 1'
                                          ))
                        # Plotting the straight line
                        fig_plane_path.add_trace(go.Scatter
                                      (x = [tang_pts_1[0, 0], tang_pts_1[1, 0]],
                                       y = [tang_pts_1[0, 1], tang_pts_1[1, 1]],
                                       mode = 'lines',
                                       line = dict(color = 'blue', width = 3),
                                       showlegend = False
                                          ))
                        # Plotting the final arc
                        fig_plane_path.add_trace(go.Scatter
                                      (x = np.array([final_circle_1[0] +\
                                                     rad_tight_turn*math.cos(i)\
                                                     for i in phi_circle_2]),
                                       y = np.array([final_circle_1[1] +\
                                                     rad_tight_turn*math.sin(i)\
                                                     for i in phi_circle_2]),
                                       mode = 'lines',
                                       line = dict(color = 'blue', width = 3),
                                       showlegend = False
                                          ))
                            
                        # Obtaining the global coordinates corresponding to the arc
                        # and the straight line on the cylinder
                        points_arc_seg_global_1,_ = mapping_arc(np.array([0, 0]), tang_pts_1[0, :],\
                                                                 initial_circle, rad_tight_turn,\
                                                                 path_types[i][0], R, ini_pos)
                        points_line_seg_global_1,_ = mapping_line_segment(tang_pts_1[0, :], tang_pts_1[1, :],\
                                                                          R, ini_pos)
                        points_arc_seg_global_2,_ = mapping_arc(tang_pts_1[1, :], pos_plane_1,\
                                                                  final_circle_1, rad_tight_turn,\
                                                                  path_types[i][2], R, ini_pos)
                        points_global = np.concatenate((np.concatenate((points_arc_seg_global_1,\
                                                        points_line_seg_global_1)), points_arc_seg_global_2))
                        fig_cylinder_path.add_trace(go.Scatter3d
                                                    (x = points_global[:, 0],
                                                     y = points_global[:, 1],
                                                     z = points_global[:, 2],
                                                     mode = 'lines',
                                                     line = dict(color = 'blue', width = 5, colorscale = 'Viridis'),
                                                     name = 'Path 1'
                                                     ))
                    
                    # Plotting for CCC path
                    else:
                        
                        print('Counter for lrl is', count_lrl)
                        print('Counter for rlr is', count_rlr)
                        
                        # Creating range of angles depending on whether first CCC
                        # path or second CCC path is being plotted
                        if (path_types[i][0].lower() == 'l' and count_lrl == 1) or \
                            (path_types[i][0].lower() == 'r' and count_rlr == 1):
                            
                            print('First CCC path')    
                            
                            # First CCC path
                            # Angle about the first circle
                            phi_circle_1 = np.linspace(phi0, phi0 + angle_11_1, 100)
                            # Angle about the last circle
                            phi_circle_3 = np.linspace(phi2 - angle_31_1, phi2, 100)
                            
                            print(phi0 + angle_11_1)
                            
                            # Obtaining angle of first tangent point wrt center of the middle circle
                            phi1 = math.atan2(tang_pt_11_1[1] - middle_circle_1_1[1],\
                                              tang_pt_11_1[0] - middle_circle_1_1[0])
                            # Range of angles about middle circle
                            phi_circle_2 = np.linspace(phi1, phi1 + angle_21_1, 100)
                        
                        else:
                            
                            print('Second CCC path')
                            # Second CCC path
                            # Angle about the first circle
                            phi_circle_1 = np.linspace(phi0, phi0 + angle_12_1, 100)
                            # Angle about the last circle
                            phi_circle_3 = np.linspace(phi2 - angle_32_1, phi2, 100)
                            
                            print(phi0 + angle_12_1)
                            
                            # Obtaining angle of first tangent point wrt center of the middle circle
                            phi1 = math.atan2(tang_pt_12_1[1] - middle_circle_2_1[1],\
                                              tang_pt_12_1[0] - middle_circle_2_1[0])
                            # Range of angles about middle circle
                            phi_circle_2 = np.linspace(phi1, phi1 + angle_22_1, 100)
                        
                        # Plotting the first arc
                        fig_plane_path.add_trace(go.Scatter
                                      (x = np.array([initial_circle[0] +\
                                                     rad_tight_turn*math.cos(i)\
                                                     for i in phi_circle_1]),
                                       y = np.array([initial_circle[1] +\
                                                     rad_tight_turn*math.sin(i)\
                                                     for i in phi_circle_1]),
                                       mode = 'lines',
                                       line = dict(color = 'blue', width = 3),
                                       name = 'Path 1'
                                          ))
                        # Plotting the middle arc
                        if (path_types[i][0].lower() == 'l' and count_lrl == 1) or \
                            (path_types[i][0].lower() == 'r' and count_rlr == 1):
                                
                            fig_plane_path.add_trace(go.Scatter
                                      (x = np.array([middle_circle_1_1[0] +\
                                                     rad_tight_turn*math.cos(i)\
                                                     for i in phi_circle_2]),
                                       y = np.array([middle_circle_1_1[1] +\
                                                     rad_tight_turn*math.sin(i)\
                                                     for i in phi_circle_2]),
                                       mode = 'lines',
                                       line = dict(color = 'blue', width = 3),
                                       showlegend = False
                                          ))
                                    
                        else:
                            
                            fig_plane_path.add_trace(go.Scatter
                                      (x = np.array([middle_circle_2_1[0] +\
                                                     rad_tight_turn*math.cos(i)\
                                                     for i in phi_circle_2]),
                                       y = np.array([middle_circle_2_1[1] +\
                                                     rad_tight_turn*math.sin(i)\
                                                     for i in phi_circle_2]),
                                       mode = 'lines',
                                       line = dict(color = 'blue', width = 3),
                                       showlegend = False
                                          ))
                                
                        # Plotting the final arc
                        fig_plane_path.add_trace(go.Scatter
                                      (x = np.array([final_circle_1[0] +\
                                                     rad_tight_turn*math.cos(i)\
                                                     for i in phi_circle_3]),
                                       y = np.array([final_circle_1[1] +\
                                                     rad_tight_turn*math.sin(i)\
                                                     for i in phi_circle_3]),
                                       mode = 'lines',
                                       line = dict(color = 'blue', width = 3),
                                       showlegend = False
                                          ))
                            
                        # Obtaining the global coordinates corresponding to the arcs
                        # on the cylinder
                            
                        if (path_types[i][0].lower() == 'l' and count_lrl == 1) or \
                            (path_types[i][0].lower() == 'r' and count_rlr == 1):
                                
                            points_arc_seg_global_1,_ = mapping_arc(np.array([0, 0]), tang_pt_11_1,\
                                                                    initial_circle, rad_tight_turn,\
                                                                    path_types[i][0], R, ini_pos)
                            points_arc_seg_global_2,_ = mapping_arc(tang_pt_11_1, tang_pt_31_1,\
                                                                    middle_circle_1_1, rad_tight_turn,\
                                                                    path_types[i][1], R, ini_pos)
                            points_arc_seg_global_3,_ = mapping_arc(tang_pt_31_1, pos_plane_1,\
                                                                    final_circle_1, rad_tight_turn,\
                                                                    path_types[i][2], R, ini_pos)
                                
                        else:
                            
                            points_arc_seg_global_1,_ = mapping_arc(np.array([0, 0]), tang_pt_12_1,\
                                                                    initial_circle, rad_tight_turn,\
                                                                    path_types[i][0], R, ini_pos)
                            points_arc_seg_global_2,_ = mapping_arc(tang_pt_12_1, tang_pt_32_1,\
                                                                    middle_circle_2_1, rad_tight_turn,\
                                                                    path_types[i][1], R, ini_pos)
                            points_arc_seg_global_3,_ = mapping_arc(tang_pt_32_1, pos_plane_1,\
                                                                    final_circle_1, rad_tight_turn,\
                                                                    path_types[i][2], R, ini_pos)
                                
                        points_global = np.concatenate((np.concatenate((points_arc_seg_global_1,\
                                                        points_arc_seg_global_2)), points_arc_seg_global_3))
                        fig_cylinder_path.add_trace(go.Scatter3d
                                                    (x = points_global[:, 0],
                                                     y = points_global[:, 1],
                                                     z = points_global[:, 2],
                                                     mode = 'lines',
                                                     line = dict(color = 'blue', width = 5, colorscale = 'Viridis'),
                                                     name = 'Path 1'
                                                     ))
                        
                # If path exists to the second image from the initial configuration
                if np.isnan(path_lengths_img_2[i]) == False:
                
                    # Initial configuration's angle wrt initial tight circle
                    phi0 = math.atan2(ini_config[1] - initial_circle[1],\
                                      ini_config[0] - initial_circle[0])
                    # Final configuration's angle wrt initial tight circle
                    phi2 = math.atan2(fin_config_2[1] - final_circle_2[1],\
                                      fin_config_2[0] - final_circle_2[0])    
                
                    # Plotting for CSC path
                    if path_types[i][1] == 's':
                        
                        # Creating the ranges for the angles
                        phi_circle_1 = np.linspace(phi0, phi0 + angle_1_2, 100)
                        phi_circle_2 = np.linspace(phi2 - angle_2_2, phi2, 100)
                
                        # Plotting the first arc
                        fig_plane_path.add_trace(go.Scatter
                                      (x = np.array([initial_circle[0] +\
                                                     rad_tight_turn*math.cos(i)\
                                                     for i in phi_circle_1]),
                                       y = np.array([initial_circle[1] +\
                                                     rad_tight_turn*math.sin(i)\
                                                     for i in phi_circle_1]),
                                       mode = 'lines',
                                       line = dict(color = 'purple', width = 3),
                                       name = 'Path 2'
                                          ))
                        # Plotting the straight line
                        fig_plane_path.add_trace(go.Scatter
                                      (x = [tang_pts_2[0, 0], tang_pts_2[1, 0]],
                                       y = [tang_pts_2[0, 1], tang_pts_2[1, 1]],
                                       mode = 'lines',
                                       line = dict(color = 'purple', width = 3),
                                       showlegend = False
                                          ))
                        # Plotting the final arc
                        fig_plane_path.add_trace(go.Scatter
                                      (x = np.array([final_circle_2[0] +\
                                                     rad_tight_turn*math.cos(i)\
                                                     for i in phi_circle_2]),
                                       y = np.array([final_circle_2[1] +\
                                                     rad_tight_turn*math.sin(i)\
                                                     for i in phi_circle_2]),
                                       mode = 'lines',
                                       line = dict(color = 'purple', width = 3),
                                       showlegend = False
                                          ))
                            
                        # Obtaining the global coordinates corresponding to the arc
                        # and the straight line on the cylinder
                        points_arc_seg_global_1,_ = mapping_arc(np.array([0, 0]), tang_pts_2[0, :],\
                                                                 initial_circle, rad_tight_turn,\
                                                                 path_types[i][0], R, ini_pos)
                        points_line_seg_global_1,_ = mapping_line_segment(tang_pts_2[0, :], tang_pts_2[1, :],\
                                                                          R, ini_pos)
                        points_arc_seg_global_2,_ = mapping_arc(tang_pts_2[1, :], pos_plane_2,\
                                                                  final_circle_2, rad_tight_turn,\
                                                                  path_types[i][2], R, ini_pos)
                        points_global = np.concatenate((np.concatenate((points_arc_seg_global_1,\
                                                        points_line_seg_global_1)), points_arc_seg_global_2))
                        fig_cylinder_path.add_trace(go.Scatter3d
                                                    (x = points_global[:, 0],
                                                     y = points_global[:, 1],
                                                     z = points_global[:, 2],
                                                     mode = 'lines',
                                                     line = dict(color = 'purple', width = 5, colorscale = 'Viridis'),
                                                     name = 'Path 2'
                                                     ))
                
                    # Plotting for CCC path
                    else:
                        
                        print('Counter for lrl is', count_lrl)
                        print('Counter for rlr is', count_rlr)
                        
                        # Creating range of angles depending on whether first CCC
                        # path or second CCC path is being plotted
                        if (path_types[i][0].lower() == 'l' and count_lrl == 1) or \
                            (path_types[i][0].lower() == 'r' and count_rlr == 1):
                            
                            print('First CCC path')
                            
                            # First CCC path
                            # Angle about the first circle
                            phi_circle_1 = np.linspace(phi0, phi0 + angle_11_2, 100)
                            # Angle about the last circle
                            phi_circle_3 = np.linspace(phi2 - angle_31_2, phi2, 100)
                            
                            print(phi0 + angle_11_2)
                            
                            # Obtaining angle of first tangent point wrt center of the middle circle
                            phi1 = math.atan2(tang_pt_11_2[1] - middle_circle_1_2[1],\
                                              tang_pt_11_2[0] - middle_circle_1_2[0])
                            # Range of angles about middle circle
                            phi_circle_2 = np.linspace(phi1, phi1 + angle_21_2, 100)
                        
                        else:
                            
                            print('Second CCC path')
                            
                            # Second CCC path
                            # Angle about the first circle
                            phi_circle_1 = np.linspace(phi0, phi0 + angle_12_2, 100)
                            # Angle about the last circle
                            phi_circle_3 = np.linspace(phi2 - angle_32_2, phi2, 100)
                            
                            print(phi0 + angle_12_2)
                            
                            # Obtaining angle of first tangent point wrt center of the middle circle
                            phi1 = math.atan2(tang_pt_12_2[1] - middle_circle_2_2[1],\
                                              tang_pt_12_2[0] - middle_circle_2_2[0])
                            # Range of angles about middle circle
                            phi_circle_2 = np.linspace(phi1, phi1 + angle_22_2, 100)
                        
                        # Plotting the first arc
                        fig_plane_path.add_trace(go.Scatter
                                      (x = np.array([initial_circle[0] +\
                                                     rad_tight_turn*math.cos(i)\
                                                     for i in phi_circle_1]),
                                       y = np.array([initial_circle[1] +\
                                                     rad_tight_turn*math.sin(i)\
                                                     for i in phi_circle_1]),
                                       mode = 'lines',
                                       line = dict(color = 'purple', width = 3),
                                       name = 'Path 2'
                                          ))
                        # Plotting the middle arc
                        if (path_types[i][0].lower() == 'l' and count_lrl == 1) or \
                            (path_types[i][0].lower() == 'r' and count_rlr == 1):
                                
                            fig_plane_path.add_trace(go.Scatter
                                      (x = np.array([middle_circle_1_2[0] +\
                                                     rad_tight_turn*math.cos(i)\
                                                     for i in phi_circle_2]),
                                       y = np.array([middle_circle_1_2[1] +\
                                                     rad_tight_turn*math.sin(i)\
                                                     for i in phi_circle_2]),
                                       mode = 'lines',
                                       line = dict(color = 'purple', width = 3),
                                       showlegend = False
                                          ))
                                    
                        else:
                            
                            fig_plane_path.add_trace(go.Scatter
                                      (x = np.array([middle_circle_2_2[0] +\
                                                     rad_tight_turn*math.cos(i)\
                                                     for i in phi_circle_2]),
                                       y = np.array([middle_circle_2_2[1] +\
                                                     rad_tight_turn*math.sin(i)\
                                                     for i in phi_circle_2]),
                                       mode = 'lines',
                                       line = dict(color = 'purple', width = 3),
                                       showlegend = False
                                          ))
                                
                        # Plotting the final arc
                        fig_plane_path.add_trace(go.Scatter
                                      (x = np.array([final_circle_2[0] +\
                                                     rad_tight_turn*math.cos(i)\
                                                     for i in phi_circle_3]),
                                       y = np.array([final_circle_2[1] +\
                                                     rad_tight_turn*math.sin(i)\
                                                     for i in phi_circle_3]),
                                       mode = 'lines',
                                       line = dict(color = 'purple', width = 3),
                                       showlegend = False
                                          ))
                            
                        # Obtaining the global coordinates corresponding to the arcs
                        # on the cylinder
                            
                        if (path_types[i][0].lower() == 'l' and count_lrl == 1) or \
                            (path_types[i][0].lower() == 'r' and count_rlr == 1):
                                
                            points_arc_seg_global_1,_ = mapping_arc(np.array([0, 0]), tang_pt_11_2,\
                                                                    initial_circle, rad_tight_turn,\
                                                                    path_types[i][0], R, ini_pos)                            
                            points_arc_seg_global_2,_ = mapping_arc(tang_pt_11_2, tang_pt_31_2,\
                                                                    middle_circle_1_2, rad_tight_turn,\
                                                                    path_types[i][1], R, ini_pos)
                            points_arc_seg_global_3,_ = mapping_arc(tang_pt_31_2, pos_plane_2,\
                                                                    final_circle_2, rad_tight_turn,\
                                                                    path_types[i][2], R, ini_pos)
                                
                        else:
                            
                            points_arc_seg_global_1,_ = mapping_arc(np.array([0, 0]), tang_pt_12_2,\
                                                                    initial_circle, rad_tight_turn,\
                                                                    path_types[i][0], R, ini_pos)
                            points_arc_seg_global_2,_ = mapping_arc(tang_pt_12_2, tang_pt_32_2,\
                                                                    middle_circle_2_2, rad_tight_turn,\
                                                                    path_types[i][1], R, ini_pos)
                            points_arc_seg_global_3,_ = mapping_arc(tang_pt_32_2, pos_plane_2,\
                                                                    final_circle_2, rad_tight_turn,\
                                                                    path_types[i][2], R, ini_pos)
                                
                        points_global = np.concatenate((np.concatenate((points_arc_seg_global_1,\
                                                        points_arc_seg_global_2)), points_arc_seg_global_3))
                        fig_cylinder_path.add_trace(go.Scatter3d
                                                    (x = points_global[:, 0],
                                                     y = points_global[:, 1],
                                                     z = points_global[:, 2],
                                                     mode = 'lines',
                                                     line = dict(color = 'purple', width = 5, colorscale = 'Viridis'),
                                                     name = 'Path 2'
                                                     ))
                
                fig_plane_path.update_layout(xaxis_title = 'x (m)',
                                      yaxis_title = 'z (m)',
                                      xaxis_range = [-2*(math.pi*R + rad_tight_turn),\
                                                     2*(math.pi*R + rad_tight_turn)])
                    
                fig_cylinder_path.update_layout(
                    # width = 800,
                    # height = 700,
                    # autosize = False,
                    scene = dict(
                        camera = dict(
                            up = dict(
                                x = 0,
                                y = 0,
                                z = 1
                            ),
                            eye=dict(
                                # x = 0,
                                # y = 1.0707,
                                # z = 1,
                                x = 1.2,
                                y = 1.0707,
                                z = 1
                            )
                        ),
                        aspectratio = dict(x = 0.75, y = 0.75, z = 0.5),
                        aspectmode = 'manual',
                        xaxis_title = 'X (m)',
                        yaxis_title = 'Y (m)',
                        zaxis_title = 'Z (m)'
                    ))
                
                if path_types[i][1].lower() == 's':
                    
                    fig_plane_path.update_layout(title_text = path_types[i].upper() +\
                                                 ' path between initial and final configurations on the plane')
                    fig_cylinder_path.update_layout(title_text = path_types[i].upper() +\
                                                    ' path between initial and final configurations on the cylinder')

                elif (path_types[i][0:-1].lower() == 'lr' and count_lrl == 1) or\
                    (path_types[i][0:-1].lower() == 'rl' and count_rlr == 1):

                    fig_plane_path.update_layout(title_text = 'First ' + path_types[i].upper() +\
                                                 ' path between initial and final configurations on the plane')
                    fig_cylinder_path.update_layout(title_text = 'First ' + path_types[i].upper() +\
                                                    ' path between initial and final configurations on the cylinder')

                else:

                    fig_plane_path.update_layout(title_text = 'Second ' + path_types[i].upper() +\
                                                 ' path between initial and final configurations on the plane')
                    fig_cylinder_path.update_layout(title_text = 'Second ' + path_types[i].upper() +\
                                                    ' path between initial and final configurations on the cylinder')
                            
                # Updating the counter depending on path type if CCC path type
                if path_types[i][0:-1].lower() == 'lr':
                    
                    count_lrl += 1
                    
                elif path_types[i][0:-1].lower() == 'rl':
                    
                    count_rlr += 1
                
                with open(filename, 'a') as f:
                    f.write(fig_plane_path.to_html(full_html=False, include_plotlyjs='cdn'))
                
                with open(filename, 'a') as f:
                    f.write(fig_cylinder_path.to_html(full_html=False, include_plotlyjs='cdn'))        