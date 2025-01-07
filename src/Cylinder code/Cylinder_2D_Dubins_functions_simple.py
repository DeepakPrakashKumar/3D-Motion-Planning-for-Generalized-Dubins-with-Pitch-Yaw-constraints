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
import sys

# Including the following command to ensure that python is able to find the relevant files afer changing directory
sys.path.insert(0, '')

# Loading the plotting class
from plotting_class import plotting_functions

# Obtaining the current directory
cwd = os.getcwd()
# Changing to one directory higher (since planar Dubins functions is located in another folder)
os.chdir("..")
# Obtaining current common directory
common_dir = os.getcwd()
# Loading the 2D Dubins code
rel_path_codes = '\Plane code'
# Changing the directory
os.chdir(common_dir + rel_path_codes)
# Importing the functions
from Plane_Dubins_functions import points_path, optimal_dubins_path
    
# Returning to the original directory
os.chdir(cwd)

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
        
    elif deltheta == 0: # HERE, CONSIDER TWO ADDITIONAL ANGLES; -2*PI, AND 2*PI.
        
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
        The default is 1. If value is 0, optimal path between the initial
        and final configuration is returned, but plots are not generated. If 1,
        the plots are visualized in the passed html file.
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
    path_length_img_1, path_params_img_1, path_type_img_1, _, _ = \
        optimal_dubins_path(ini_config, fin_config_1, rad_tight_turn, False)
    path_length_img_2, path_params_img_2, path_type_img_2, _, _ = \
        optimal_dubins_path(ini_config, fin_config_2, rad_tight_turn, False)
    
    # Finding the optimal path type, the minimum length of the path, and the parameters
    # of the path
    if path_length_img_1 <= path_length_img_2:
        
        min_path_length = path_length_img_1
        # Obtaining the parameters of the optimal path
        t_opt = path_params_img_1[0]
        p_opt = path_params_img_1[1]
        q_opt = path_params_img_1[2]
        opt_fin_img = fin_config_1
        opt_path_type = path_type_img_1
        
    else:
        
        min_path_length = path_length_img_2
        t_opt = path_params_img_2[0]
        p_opt = path_params_img_2[1]
        q_opt = path_params_img_2[2]
        opt_fin_img = fin_config_2
        opt_path_type = path_type_img_2
    
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
        optimal_dubins_path(ini_config, fin_config_1, rad_tight_turn, filename)
        with open(filename, 'a') as f:
            f.write("-----------Paths to second image of the final configuration-----------")
        optimal_dubins_path(ini_config, fin_config_2, rad_tight_turn, filename)
        with open(filename, 'a') as f:
            f.write("Optimal path is of type " + opt_path_type.upper() + " and of length "\
                    + str(min_path_length) + ".<br />")
                
        # Generating each path on the plane and the corresponding path on the cylinder            
        # Making a copy of the figure on the plane to augment the path to
        fig_plane_path = copy.deepcopy(fig_plane)
        # Making a copy of the figure on the cylinder to augment the path to
        fig_cylinder_path = copy.deepcopy(fig_cylinder)   
        
        # Obtaining the points along the path using the points_path function if path exists
        # Path to first image        
        # Obtaining the points along the path for the first image and plotting on the plane
        pts_path_plane_x_coord_img_1, pts_path_plane_y_coord_img_1 =\
            points_path(ini_config, rad_tight_turn, path_params_img_1, path_type_img_1)
        fig_plane_path.scatter_2D(pts_path_plane_x_coord_img_1, pts_path_plane_y_coord_img_1,\
                                    'blue', 'Optimal path to first image of type ' + path_type_img_1.upper())
        
        # Path to second image        
        # Obtaining the points along the path for the second image and plotting on the plane
        pts_path_plane_x_coord_img_2, pts_path_plane_y_coord_img_2 =\
            points_path(ini_config, rad_tight_turn, path_params_img_2, path_type_img_2)
        fig_plane_path.scatter_2D(pts_path_plane_x_coord_img_2, pts_path_plane_y_coord_img_2,\
                                    'purple', 'Optimal path to second image of type ' + path_type_img_2.upper())
        
        # Generating the paths on the cylinder using the coordinates of the curve on the plane
        # Using the coordinates of points on the path on the plane, the corresponding
        # coordinates of the path on the cylinder in the global frame can be directly obtained.
        # The same expression is used to plot the path on the cylinder
        
        # Obtaining the coordinates for the path to the first image of the final configuration
        # that is wrapped onto the cylinder if path exists        
        path_img_1_x_coord = np.array([R*math.cos(ini_pos_parametrization_angle + (j)/R)\
                                        for j in pts_path_plane_x_coord_img_1])
        path_img_1_y_coord = np.array([R*math.sin(ini_pos_parametrization_angle + (j)/R)\
                                        for j in pts_path_plane_x_coord_img_1])
        path_img_1_z_coord = np.array([(ini_pos[2] + j) for j in pts_path_plane_y_coord_img_1])
        # Plotting the path on the cylinder
        fig_cylinder_path.scatter_3D(path_img_1_x_coord, path_img_1_y_coord,\
                                        path_img_1_z_coord, 'blue', 'Optimal path to the first image of type '\
                                             + path_type_img_1.upper())
        
        # Obtaining the coordinates for the path to the second image of the final configuration
        # that is wrapped onto the cylinder if path exists            
        path_img_2_x_coord = np.array([R*math.cos(ini_pos_parametrization_angle + (j)/R)\
                                        for j in pts_path_plane_x_coord_img_2])
        path_img_2_y_coord = np.array([R*math.sin(ini_pos_parametrization_angle + (j)/R)\
                                        for j in pts_path_plane_x_coord_img_2])
        path_img_2_z_coord = np.array([(ini_pos[2] + j) for j in pts_path_plane_y_coord_img_2])
        # Plotting the path on the cylinder
        fig_cylinder_path.scatter_3D(path_img_2_x_coord, path_img_2_y_coord,\
                                        path_img_2_z_coord, 'purple', 'Optimal path to the second image of type '\
                                              + path_type_img_2.upper())
        
        # Adding the figures, i.e., on the plane and the cylinder to the html file        
        # Adding labels to the axis and title to the plot
        fig_plane_path.update_layout_2D('x (m)', [-2*(math.pi*R + rad_tight_turn),\
                                                    2*(math.pi*R + rad_tight_turn)],\
                                        'z (m)', [-zmax-2*rad_tight_turn, zmax+2*rad_tight_turn],\
                                        'Optimal paths on the plane')
        # Writing onto the html file
        fig_plane_path.writing_fig_to_html(filename, 'a')
            
        # Adding labels to the axis and title to the plot
        fig_cylinder_path.update_layout_3D('X (m)', 'Y (m)', 'Z (m)',\
                                            'Optimal paths on the cylinder')
        # Writing onto the html file
        fig_cylinder_path.writing_fig_to_html(filename, 'a')
    
    # Generating the coordinates of the optimal path
    # Obtaining the points for the optimal path on the plane
    pts_opt_path_plane_x_coord, pts_opt_path_plane_y_coord =\
        points_path(ini_config, rad_tight_turn, [t_opt, p_opt, q_opt], opt_path_type)
            
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
    
    return min_path_length, opt_path_type, points_path_global