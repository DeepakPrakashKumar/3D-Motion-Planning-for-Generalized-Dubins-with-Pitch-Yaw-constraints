# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 20:51:56 2022

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
from Dubins_functions import CCC_path_efficient, CSC_path_efficient, points_path,\
    tight_circles, Seg_pts
    
# Returning to the original directory
path = 'D:\TAMU\Research\Cylinder code'

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

def unwrapped_configurations_2D(ini_pos, ini_tang_vect, R, point_pos, point_tang_vect, zmax = 20):
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
        Two points are possible due to periodicity of theta, the angle of the final configuration
        wrt to the initial configuration on the profile of the cylinder. If theta is less than 0,
        then the two images of the final configuration correspond to theta and theta + 2*pi. If theta
        is greater than 0, then the two images of the final configuration correspond to theta - 2*pi
        and theta.
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
    # in the global frame - this angle theta corresponds to the location on the
    # profile of the cylinder. That is, a point on the profile of the cylinder
    # is parameterized using theta as (R cos(theta), R sin (theta), z).
    thetai = math.atan2(ini_pos[1], ini_pos[0])
    thetao = math.atan2(point_pos[1], point_pos[0])
    
    # Angle of point with respect to the initial point and defining it in the range
    # -pi to pi
    deltheta = thetao - thetai
    deltheta = math.atan2(math.sin(deltheta), math.cos(deltheta)) # using atan2
    # function to make sure that the angle is in the range -pi to pi.
    
    # Finding the angles of the two images of the final configuration on the unwrapping
    # plane so as to accordingly find the positions of the two images of the final 
    # configuration on the unwrapping plane using these angles
    if deltheta < 0:
        
        theta1 = deltheta + 2*math.pi
        theta2 = deltheta
        
    elif deltheta > 0:
        
        theta1 = deltheta
        theta2 = deltheta - 2*math.pi
        
    elif deltheta == 0:
        
        theta1 = theta2 = deltheta
        
    # Coordinates of the final position on the plane depending on how unwrapping is done.
    # Mapping is done to (R theta, delta_z), where delta_z is the difference in the
    # vertical distance.
    pos_plane_1 = np.array([R*theta1, point_pos[2] - ini_pos[2]])
    pos_plane_2 = np.array([R*theta2, point_pos[2] - ini_pos[2]])
    
    # Heading at the initial position and the given position
    heading_ini_pos_plane = math.atan2(ini_tang_vect[2],\
                                       (-ini_tang_vect[0]*math.sin(thetai)\
                                        + ini_tang_vect[1]*math.cos(thetai)))
    heading_point_pos_plane = math.atan2(point_tang_vect[2],\
                                         (-point_tang_vect[0]*math.sin(thetao)\
                                          + point_tang_vect[1]*math.cos(thetao)))
        
    return pos_plane_1, pos_plane_2, heading_ini_pos_plane, heading_point_pos_plane

def Dubins_path_plane(ini_config, fin_config, rad_tight_turn, filename = False):
    '''
    This function generates the length of each three segment path connecting the
    initial and final configuration on a plane. The functions developed in the file
    Dubins_functions.py is used for this purpose. The details of the generated paths
    are written onto a html file if the argument "filename" is passed. The function also
    returns the parameters for each path type, i.e., the lengths/angles of the first,
    second, and third arcs. In this function, t corresponds to the angle of the first
    segment, p corresponds to the angle (for CCC path) or length (for CSC path) for the
    second segment, and q corresponds to the angle of the third segment.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the x and y coordinate (first two indeces) of the initial position and
        the heading angle (last entry).
    fin_config : Numpy 1x3 array
        Contains the x and y coordinate (first two indeces) of the final position and
        the heading angle (last entry).
    rad_tight_turn : Scalar
        Radius of the tight circle turn.
    filename : String
        Name of the html file in which the results should be stored.

    Returns
    -------
    path_types : Numpy 1x8 array
        Contains strings representing the path type.
    path_lengths : Numpy 1x8 array
        Contains the path length corresponding to each path type in the path_types variable.
        NaN is returned when a corresponding path does not exist.
    t, p, q : Numpy 1x8 arrays
        Contains the angles (for tight turns) and lengths (for straight line segments) for
        each path type.
    '''
    
    path_types = np.array(['lsl', 'rsr', 'lsr', 'rsl', 'lrl', 'lrl', 'rlr', 'rlr'])
    path_lengths = np.empty(8)
    t = np.empty(8)
    p = np.empty(8)
    q = np.empty(8)

    count_lrl = 1 # counter to ensure lrl path is run only once, as the CCC path generation
    # function generates both lrl paths
    count_rlr = 1 # counter to ensure rlr path is run only once
    
    for i in range(len(path_types)):
        
        # Checking if path is CSC type
        if path_types[i][1] == 's':
        
            # Obtaining the path length and lengths of the segments
            path_lengths[i], t[i], p[i], q[i] = CSC_path_efficient(ini_config, fin_config,\
                                                                   rad_tight_turn, path_types[i], 0)
            
        # If not, path is of type CCC
        else:
            
            if (path_types[i][0] == 'l' and count_lrl == 1) or\
                (path_types[i][0] == 'r' and count_rlr == 1):
            
                # Obtaining the path length and lengths of the segments and adding to the array
                path_lengths[i: i+2], t[i: i+2], p[i: i+2], q[i: i+2] =\
                    CCC_path_efficient(ini_config, fin_config, rad_tight_turn, path_types[i], 0)
                # NOTE: If the CCC path does not exist, np.NaN is added for both the "paths" in the
                # two indeces for each variable.
                
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
    
    return path_types, path_lengths, t, p, q

def generate_visualize_path(ini_pos, ini_tang_vect, R, fin_pos, fin_tang_vect,\
                            zmax = 20, visualization = 1, rad_tight_turn = 1, filename = 'temp.html'):
    '''
    This function generates a path between two points on a right circular cylinder
    using the 2D Dubins result and visualizes the path. Note that the right circular
    cylinder is considered to be such that the axis of the cylinder is along the global
    z-axis, and its center coincides with the origin of the global frame.

    Parameters
    ----------
    ini_pos : Numpy 1x3 array
        Contains the coordinates of the initial position on the cylinder.
    ini_tang_vect : Numpy 1x3 array
        Contains the direction cosines of the tangent vector at the coordinates
        given by ini_pos. The direction cosines are given in the global frame.
    R : Scalar
        Radius of the cylinder.
    fin_pos : Numpy 1x3 array
        Coordinates corresponding to a point that should be reached.
    fin_tang_vect : Numpy 1x3 array
        Direction cosines of the random tangent vector at point_pos.
    zmax: Scalar
        Maximum value corresponding to the z-value of the randomly generated coordinates.
    visualization : TYPE, optional
        DESCRIPTION. The default is 1. If value is 0, optimal path between the initial
        and final configuration is returned, but plots are not generated.
    rad_tight_turn : Scalar
        Radius of the tight circle turn.

    Returns
    -------
    min_path_length : Scalar
        Contains the length of the optimal path from the initial to the final configuration
        on the cylinder.
    path_types[min_index] : String
        Contains the type of the optimal path.
    points_path_global : Numpy array
        Contains the coordinates of points along the optimal path.

    '''
    
    # Transforming the configurations to the unwrapping plane that is defined using the body
    # frame fixed at the initial configuration
    fin_pos_plane_1, fin_pos_plane_2, heading_ini_pos_plane, heading_fin_pos_plane = \
        unwrapped_configurations_2D(ini_pos, ini_tang_vect, R, fin_pos, fin_tang_vect, zmax)
        
    # Generating the initial and final configurations on the unwrapping plane to pass to
    # the 2-D Dubins functions
    ini_config = np.array([0, 0, heading_ini_pos_plane])
    # Image 1 of the final configuration on the plane
    fin_config_1 = np.array([fin_pos_plane_1[0], fin_pos_plane_1[1], heading_fin_pos_plane])
    # Image 2 of the final configuration on the plane
    fin_config_2 = np.array([fin_pos_plane_2[0], fin_pos_plane_2[1], heading_fin_pos_plane])
    
    # Obtaining all the information regarding the paths from the initial configuration to
    # the two images of the final configuration
    path_types, path_lengths_img_1, t_paths_img_1, p_paths_img_1, q_paths_img_1 =\
        Dubins_path_plane(ini_config, fin_config_1, rad_tight_turn, False)
    _, path_lengths_img_2, t_paths_img_2, p_paths_img_2, q_paths_img_2 =\
        Dubins_path_plane(ini_config, fin_config_2, rad_tight_turn, False)
        
    # Finding the optimal path
    min_path_length_img1 = min(path_lengths_img_1)
    min_path_length_img2 = min(path_lengths_img_2)
    
    # Finding the optimal path type, the minimum length of the path, and the parameters
    # of the path
    if min_path_length_img1 <= min_path_length_img2:
        
        min_path_index = np.nanargmin(path_lengths_img_1)
        min_path_length = min_path_length_img1
        t_opt = t_paths_img_1[min_path_index]
        p_opt = p_paths_img_1[min_path_index]
        q_opt = q_paths_img_1[min_path_index]
        opt_fin_img = fin_config_1
        
    else:
        
        min_path_index = np.nanargmin(path_lengths_img_2)
        min_path_length = min_path_length_img2
        t_opt = t_paths_img_2[min_path_index]
        p_opt = p_paths_img_2[min_path_index]
        q_opt = q_paths_img_2[min_path_index]
        opt_fin_img = fin_config_2
    
    if visualization == 1:
        
        # Creating a 3D plot. fig_cylinder is declared as an instance of the class
        # plotting_functions.
        fig_cylinder = plotting_functions()
        
        # Plotting the cylinder
        theta = np.linspace(0, 2*math.pi, 50)
        z = np.linspace(-zmax - 2*rad_tight_turn, zmax + 2*rad_tight_turn, 50)
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
        # Adding the axes for the body frame about which the cylinder is unwrapped
        ini_pos_parametrization_angle = math.atan2(ini_pos[1], ini_pos[0]) # obtaining the angle
        # at which the initial position is present on the profile of the cylinder
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
                                   'z (m)', [-zmax-2*rad_tight_turn, zmax+2*rad_tight_turn],\
                                   'Initial and final configurations on a plane')
        # Writing onto the html file
        fig_plane.writing_fig_to_html(filename, 'a')
        
        # Obtaining the lengths of all paths to image 1 and image 2 of the final configuration
        # from the initial configuration and writing on the html file (done in the
        # Dubins_path_plane function)
        
        with open(filename, 'a') as f:
            f.write("<br\><br\><br\>Details of the paths:")
        with open(filename, 'a') as f:
            f.write("<br\>-----------Paths to first image of the final configuration-----------")
        Dubins_path_plane(ini_config, fin_config_1, rad_tight_turn, filename)
        with open(filename, 'a') as f:
            f.write("-----------Paths to second image of the final configuration-----------")
        Dubins_path_plane(ini_config, fin_config_2, rad_tight_turn, filename)
        with open(filename, 'a') as f:
            f.write("Optimal path is of type " + path_types[min_path_index].upper() + " and of length "\
                    + str(min_path_length) + ".<br />")
                
        # Generating each path on the plane and the corresponding path on the cylinder
        # Counter for LRL and RLR paths to accordingly modify the title
        count_lrl = 1
        count_rlr = 1
        for i in range(len(path_types)):
            
            # Making a copy of the figure on the plane to augment the path to
            fig_plane_path = copy.deepcopy(fig_plane)
            # Making a copy of the figure on the cylinder to augment the path to
            fig_cylinder_path = copy.deepcopy(fig_cylinder)   
            
            # Obtaining the points along the path using the points_path function if path exists
            # Path to first image
            if np.isnan(path_lengths_img_1[i]) == False:
            
                # Obtaining the points along the path for the first image and plotting on the plane
                pts_path_plane_x_coord_img_1, pts_path_plane_y_coord_img_1 =\
                    points_path(ini_config, fin_config_1, rad_tight_turn, t_paths_img_1[i],\
                                p_paths_img_1[i], q_paths_img_1[i], path_types[i])
                fig_plane_path.scatter_2D(pts_path_plane_x_coord_img_1, pts_path_plane_y_coord_img_1,\
                                          'blue', 'Path to first image')
            
            # Path to second image
            if np.isnan(path_lengths_img_2[i]) == False:
            
                # Obtaining the points along the path for the second image and plotting on the plane
                pts_path_plane_x_coord_img_2, pts_path_plane_y_coord_img_2 =\
                    points_path(ini_config, fin_config_2, rad_tight_turn, t_paths_img_2[i],\
                                p_paths_img_2[i], q_paths_img_2[i], path_types[i])
                fig_plane_path.scatter_2D(pts_path_plane_x_coord_img_2, pts_path_plane_y_coord_img_2,\
                                          'purple', 'Path to second image')       
            
            # Generating the paths on the cylinder using the coordinates of the curve on the plane
            # Using the coordinates of points on the path on the plane, the corresponding
            # coordinates of the path on the cylinder in the global frame can be directly obtained.
            # The same expression is used to plot the path on the cylinder
            
            # Obtaining the coordinates for the path to the first image of the final configuration
            # that is wrapped onto the cylinder if path exists
            if np.isnan(path_lengths_img_1[i]) == False:
            
                path_img_1_x_coord = np.array([R*math.cos(ini_pos_parametrization_angle + (j)/R)\
                                               for j in pts_path_plane_x_coord_img_1])
                path_img_1_y_coord = np.array([R*math.sin(ini_pos_parametrization_angle + (j)/R)\
                                               for j in pts_path_plane_x_coord_img_1])
                path_img_1_z_coord = np.array([(ini_pos[2] + j) for j in pts_path_plane_y_coord_img_1])
                # Plotting the path on the cylinder
                fig_cylinder_path.scatter_3D(path_img_1_x_coord, path_img_1_y_coord,\
                                             path_img_1_z_coord, 'blue', 'Path to the first image')
            
            # Obtaining the coordinates for the path to the second image of the final configuration
            # that is wrapped onto the cylinder if path exists
            if np.isnan(path_lengths_img_2[i]) == False:
                
                path_img_2_x_coord = np.array([R*math.cos(ini_pos_parametrization_angle + (j)/R)\
                                               for j in pts_path_plane_x_coord_img_2])
                path_img_2_y_coord = np.array([R*math.sin(ini_pos_parametrization_angle + (j)/R)\
                                               for j in pts_path_plane_x_coord_img_2])
                path_img_2_z_coord = np.array([(ini_pos[2] + j) for j in pts_path_plane_y_coord_img_2])
                # Plotting the path on the cylinder
                fig_cylinder_path.scatter_3D(path_img_2_x_coord, path_img_2_y_coord,\
                                             path_img_2_z_coord, 'purple', 'Path to the second image')
            
            # Adding the figures, i.e., on the plane and the cylinder to the html file if a path
            # to one of the two images exist
            if np.isnan(path_lengths_img_1[i]) == False or np.isnan(path_lengths_img_2[i]) == False:
            
                if path_types[i][1] == 's':
                    
                    path_name = path_types[i].upper() + ' path'
                    
                else:
                    
                    if path_types[i][1] == 'r' and count_lrl == 1:
                        
                        path_name = 'First ' + path_types[i].upper() + ' path'
                        count_lrl += 1 # Incrementing value of count_lrl
                        
                    elif path_types[i][1] == 'r' and count_lrl == 2:
                        
                        path_name = 'Second ' + path_types[i].upper() + ' path'
                        
                    elif path_types[i][1] == 'l' and count_rlr == 1:
                        
                        path_name = 'First ' + path_types[i].upper() + ' path'
                        count_rlr += 1 # Incrementing value of count_rlr
                        
                    elif path_types[i][1] == 'l' and count_rlr == 2:
                        
                        path_name = 'Second ' + path_types[i].upper() + ' path'
            
                # Adding labels to the axis and title to the plot
                fig_plane_path.update_layout_2D('x (m)', [-2*(math.pi*R + rad_tight_turn),\
                                                          2*(math.pi*R + rad_tight_turn)],\
                                                'z (m)', [-zmax-2*rad_tight_turn, zmax+2*rad_tight_turn],\
                                                path_name + ' on the plane')
                # Writing onto the html file
                fig_plane_path.writing_fig_to_html(filename, 'a')
                    
                # Adding labels to the axis and title to the plot
                fig_cylinder_path.update_layout_3D('X (m)', 'Y (m)', 'Z (m)',\
                                                   path_name + ' on the cylinder')
                # Writing onto the html file
                fig_cylinder_path.writing_fig_to_html(filename, 'a')
    
    # Generating the coordinates of the optimal path
    # Obtaining the points for the optimal path on the plane
    pts_opt_path_plane_x_coord, pts_opt_path_plane_y_coord =\
        points_path(ini_config, opt_fin_img, rad_tight_turn, t_opt,\
                    p_opt, q_opt, path_types[min_path_index])
            
    # Obtaining the coordinates for the optimal path in the global frame for the cylinder
    points_path_global = np.empty((np.size(pts_opt_path_plane_x_coord), 3))
    
    points_path_global[:, 0] = np.array([R*math.cos(ini_pos_parametrization_angle + (j)/R)\
                                         for j in pts_opt_path_plane_x_coord])
    points_path_global[:, 1] = np.array([R*math.sin(ini_pos_parametrization_angle + (j)/R)\
                                         for j in pts_opt_path_plane_x_coord])
    points_path_global[:, 2] = np.array([(ini_pos[2] + j) for j in pts_opt_path_plane_y_coord])
    
    if visualization == 1:
        
        # Making a copy for the initial and final configurations on the cylinder
        fig_cylinder_path = copy.deepcopy(fig_cylinder)
        # Plotting the optimal path on the cylinder
        fig_cylinder_path.scatter_3D(points_path_global[:, 0], points_path_global[:, 1],\
                                     points_path_global[:, 2], 'blue', 'Optimal path')
        # Adding labels to the axis and title to the plot
        fig_cylinder_path.update_layout_3D('X (m)', 'Y (m)', 'Z (m)',\
                                           'Optimal path on the cylinder')
        # Writing onto the html file
        fig_cylinder_path.writing_fig_to_html(filename, 'a')
    
    return min_path_length, path_types[min_path_index], points_path_global