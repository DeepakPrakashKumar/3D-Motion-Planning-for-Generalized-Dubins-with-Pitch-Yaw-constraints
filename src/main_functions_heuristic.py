# -*- coding: utf-8 -*-
"""
Created on Sun Feb 6 20:51:49 2022

@author: Deepak Prakash Kumar
"""

import numpy as np
import os
import math
import copy
import sys
from pathlib import Path

# Including the following command to ensure that python is able to find the relevant files afer changing directory
sys.path.insert(0, '')
# Obtaining the current directory
cwd = os.getcwd()
current_directory = Path(__file__).parent
path_str = str(current_directory)

# Importing code for plotting
rel_path = '\Cylinder code'
os.chdir(path_str + rel_path)
from plotting_class import plotting_functions
# Importing code for the cylinder
from Cylinder_2D_Dubins_functions_simple import generate_visualize_path

# Importing code for the sphere
rel_path = '\Sphere code'
os.chdir(path_str + rel_path)
from Path_generation_sphere import optimal_path_sphere_three_seg, generate_points_sphere

# Importing code for the plane
rel_path = '\Plane code'
os.chdir(path_str + rel_path)
from Plane_Dubins_functions import optimal_dubins_path

# Returning to initial directory
os.chdir(cwd)

# Importing code for sphere-cylinder-sphere, sphere-plane-sphere, and sphere-sphere-sphere setups
from sphere_cylinder_sphere_function import *
from sphere_plane_sphere_path_function import *
from sphere_sphere_sphere_path_function import *

def generate_random_configs_3D(xlim, ylim, zlim):
    '''
    This function generates a random configuration in the 3D space given the limits
    on the x-axis, y-axis, and z-axis values

    Parameters
    ----------
    xlim, ylim, zlim : Scalars
        Limits on the x-axis, y-axis, and z-axis values for the position of the
        configuration.

    Returns
    -------
    config : Numpy 4x3 array
        Contains the position in the first row, the direction cosines of the tangent
        vector, tangent normal vector, and the surface normal vector in the second,
        third, and fourth rows, respectively.

    '''
    
    x_pos = (np.random.rand() - 0.5)*xlim # since the random number of 0.5 should correspond
    # to zero x-coordinate
    y_pos = (np.random.rand() - 0.5)*ylim
    z_pos = (np.random.rand() - 0.5)*zlim

    # Generating a random vector for the tangent vector, and normalizing it
    T = np.random.rand(3)
    T = T/np.linalg.norm(T)

    # Generating another random vector and using Gram-Schmidt to generate an orthonormal
    # vector to T
    temp = np.random.rand(3)
    # Checking that the generated random vector is not linear dependent on T
    if np.linalg.norm(temp - np.dot(temp, T)*T) >= 0.01:
        
       t =  (temp - np.dot(temp, T)*T)/np.linalg.norm(temp - np.dot(temp, T)*T)
   
    else:
       
       raise Exception('Regenerate the random vector')
   
    # Finding the surface normal vector
    u = np.cross(T, t)
    
    config = np.array([[x_pos, y_pos, z_pos], T, t, u])
    
    return config

def Dubins_3D_numerical_path_on_surfaces(ini_config, fin_config, r, R, disc_no,\
                                         visualization = 1, filename = "temp.html"):
    '''
    This function plots the initial and the final configuration, and the spheres
    and cylinders that connect the two configurations.

    Parameters
    ----------
    ini_config : Numpy 4x3 array
        Contains the initial position in the first row, the direction cosines of
        the initial tangent vector in the second row, the direction cosines of the
        initial tangent normal vector in the third row, and the direction cosines
        of the surface normal vector in the fourth row.
    fin_config : Numpy 4x3 array
        Contains the final position vector, the direction cosines of the final
        tangent vector, the tangent normal vector, and the surface normal vector
        in the same format as the ini_config variable.
    r: Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the surface.
    visualization : Scalar, optional
        Variable to decide whether to show the plot of the configurations and the
        surfaces. Default is equal to 1.
    filename : String, optional
        Name of the file in which the figure should be written. Used when visualization
        is set to 1. Default is "temp.html".

    Returns
    -------
    Numpy 4x3 array: Contains the center of the four spheres.
    Numpy 4x3 array: Contains the direction cosines of the axis corresponding to the four
        cylinders.
    Numpy 4x1 array: Contains the length of the four cylinders.

    '''
    
    print('Computing feasible paths through different surfaces.')
    # Computing the center of the inner and outer spheres at the initial and final
    # configurations    
    ini_loc_inner_sp = ini_config[0, :] + R*ini_config[3, :]
    ini_loc_outer_sp = ini_config[0, :] - R*ini_config[3, :]
    fin_loc_inner_sp = fin_config[0, :] + R*fin_config[3, :]
    fin_loc_outer_sp = fin_config[0, :] - R*fin_config[3, :]

    # Computing the axis of each cylinder connecting the pair of inner and outer spheres
    # at the initial and final configurations.
    # Obtaining the unit vector along the axis of the inner sphere connections    
    axis_ii = (fin_loc_inner_sp - ini_loc_inner_sp)\
        /np.linalg.norm(fin_loc_inner_sp - ini_loc_inner_sp)
    ht_ii = np.linalg.norm(fin_loc_inner_sp - ini_loc_inner_sp)
    # Obtaining the unit vector along the axis of the outer sphere connections
    axis_oo = (fin_loc_outer_sp - ini_loc_outer_sp)\
        /np.linalg.norm(fin_loc_outer_sp - ini_loc_outer_sp)
    ht_oo = np.linalg.norm(fin_loc_outer_sp - ini_loc_outer_sp)

    # Obtaining the line segment corresponding to the axis of the cross-tangent planes
    axis_io = (fin_loc_outer_sp - ini_loc_inner_sp)/np.linalg.norm(fin_loc_outer_sp - ini_loc_inner_sp)
    ht_io = np.linalg.norm(fin_loc_outer_sp - ini_loc_inner_sp)
    axis_oi = (fin_loc_inner_sp - ini_loc_outer_sp)/np.linalg.norm(fin_loc_inner_sp - ini_loc_outer_sp)
    ht_oi = np.linalg.norm(fin_loc_inner_sp - ini_loc_outer_sp)
    
    # Creating a plotly figure environment
    plot_figure_configs = plotting_functions()

    if visualization == 1:

        # Showing the initial and final points
        plot_figure_configs.points_3D([ini_config[0, 0]], [ini_config[0, 1]], [ini_config[0, 2]],\
                              'red', 'Initial point', 'circle')
        plot_figure_configs.points_3D([fin_config[0, 0]], [fin_config[0, 1]], [fin_config[0, 2]],\
                              'black', 'Final point', 'diamond')
        # Showing the initial and final orientations
        plot_figure_configs.arrows_3D([ini_config[0, 0]], [ini_config[0, 1]], [ini_config[0, 2]],\
                              [ini_config[1, 0]], [ini_config[1, 1]], [ini_config[1, 2]],\
                              'orange', 'oranges', 'Tangent vector', 5, 5, 4, 'n')
        plot_figure_configs.arrows_3D([ini_config[0, 0]], [ini_config[0, 1]], [ini_config[0, 2]],\
                              [ini_config[2, 0]], [ini_config[2, 1]], [ini_config[2, 2]],\
                              'purple', 'purp', 'Tangent normal vector', 5, 5, 4, 'n')
        plot_figure_configs.arrows_3D([ini_config[0, 0]], [ini_config[0, 1]], [ini_config[0, 2]],\
                              [ini_config[3, 0]], [ini_config[3, 1]], [ini_config[3, 2]],\
                              'green', 'greens', 'Surface normal vector', 5, 5, 4, 'n')
        plot_figure_configs.arrows_3D([fin_config[0, 0]], [fin_config[0, 1]], [fin_config[0, 2]],\
                              [fin_config[1, 0]], [fin_config[1, 1]], [fin_config[1, 2]],\
                              'orange', 'oranges', False, 5, 5, 4, 'n')
        plot_figure_configs.arrows_3D([fin_config[0, 0]], [fin_config[0, 1]], [fin_config[0, 2]],\
                              [fin_config[2, 0]], [fin_config[2, 1]], [fin_config[2, 2]],\
                              'purple', 'purp', False, 5, 5, 4, 'n')
        plot_figure_configs.arrows_3D([fin_config[0, 0]], [fin_config[0, 1]], [fin_config[0, 2]],\
                              [fin_config[3, 0]], [fin_config[3, 1]], [fin_config[3, 2]],\
                              'green', 'greens', False, 5, 5, 4, 'n')
        
        plot_figure_configs.update_layout_3D('X (m)', 'Y (m)', 'Z (m)', 'Visualization of configurations')
            
        # Writing the figure on the html file
        plot_figure_configs.writing_fig_to_html(filename, 'w')
        
    # # We now construct the best feasible path connecting the two inner spheres and the two outer spheres
    # print('Considering path construction through cylindrical envelope connecting pair of inner spheres.')
    # min_dist_cyc_inner, points_global_cyc_inner, tang_global_cyc_inner, tang_normal_global_cyc_inner, surf_normal_global_cyc_inner =\
    #  Path_generation_sphere_cylinder_sphere(ini_config, fin_config, ini_loc_inner_sp, fin_loc_inner_sp, r, R, axis_ii, ht_ii, disc_no,\
    #                                         plot_figure_configs, visualization, filename, 'inner', 0)
    
    print('Considering path construction through cylindrical envelope connecting pair of outer spheres.')
    min_dist_cyc_outer, points_global_cyc_outer, tang_global_cyc_outer, tang_normal_global_cyc_outer, surf_normal_global_cyc_outer =\
     Path_generation_sphere_cylinder_sphere(ini_config, fin_config, ini_loc_outer_sp, fin_loc_outer_sp, r, R, axis_oo, ht_oo, disc_no,\
                                            plot_figure_configs, visualization, filename, 'outer', 0)
        
    # # We now construct the best feasible path connecting the two cross-tangent spheres
    # print('Considering path construction through cross-tangent plane connecting inner sphere at initial location and outer sphere at final location.')
    # min_dist_plane_inner_outer, points_global_inner_outer = Path_generation_sphere_plane_sphere(ini_config, fin_config, ini_loc_inner_sp, fin_loc_outer_sp,\
    #                                        r, R, axis_io, ht_io, disc_no, plot_figure_configs, visualization, filename, 0)
    
    # print('Considering path construction through cross-tangent plane connecting outer sphere at initial location and inner sphere at final location.')
    # min_dist_plane_outer_inner, points_global_outer_inner = Path_generation_sphere_plane_sphere(ini_config, fin_config, ini_loc_outer_sp, fin_loc_inner_sp,\
    #                                        r, R, axis_oi, ht_oi, disc_no, plot_figure_configs, visualization, filename, 0)

    # # We perform the connections through three sphere setup as well.
    # print('Considering path construction connecting inner spheres at initial and final configurations through an intermediary sphere.')
    # min_dist_inner_spheres, points_global_inner_spheres = Path_generation_sphere_sphere_sphere(ini_config, fin_config, ini_loc_inner_sp, fin_loc_inner_sp,\
    #                                        r, R, axis_ii, ht_ii, disc_no, plot_figure_configs, visualization, filename, 0)
    
    # print('Considering path construction connecting outer spheres at initial and final configurations through an intermediary sphere.')
    # min_dist_outer_spheres, points_global_outer_spheres = Path_generation_sphere_sphere_sphere(ini_config, fin_config, ini_loc_outer_sp, fin_loc_outer_sp,\
    #                                        r, R, axis_oo, ht_oo, disc_no, plot_figure_configs, visualization, filename, 0)
    
    # Obtaining the minimum distance path among the considered paths
    # min_dist_path_ind = np.argmin([min_dist_cyc_inner, min_dist_cyc_outer, min_dist_plane_inner_outer,\
    #                                 min_dist_plane_outer_inner, min_dist_inner_spheres, min_dist_outer_spheres])
    
    # min_dist_path = [min_dist_cyc_inner, min_dist_cyc_outer, min_dist_plane_inner_outer,\
    #                                 min_dist_plane_outer_inner, min_dist_inner_spheres, min_dist_outer_spheres][min_dist_path_ind]
    # min_dist_path_pts = [points_global_cyc_inner, points_global_cyc_outer, points_global_inner_outer,\
    #                 points_global_outer_inner, points_global_inner_spheres, points_global_outer_spheres][min_dist_path_ind]

    # path_type = ['cyc_inner', 'cyc_outer', 'plane_inner_outer', 'plane_outer_inner', 'spheres_inner', 'spheres_outer'][min_dist_path_ind]

    # min_dist_path_ind = np.argmin([min_dist_cyc_inner, min_dist_cyc_outer])
    
    # min_dist_path = [min_dist_cyc_inner, min_dist_cyc_outer][min_dist_path_ind]
    # min_dist_path_pts = [points_global_cyc_inner, points_global_cyc_outer][min_dist_path_ind]

    # path_type = ['cyc_inner', 'cyc_outer'][min_dist_path_ind]

    # # Obtaining the tangent vector, tangent normal vector, and surface normal vector as well
    # tang_global_path = [tang_global_cyc_inner, tang_global_cyc_outer][min_dist_path_ind]
    # tang_normal_global_path = [tang_normal_global_cyc_inner, tang_normal_global_cyc_outer][min_dist_path_ind]
    # surf_normal_global_path = [surf_normal_global_cyc_inner, surf_normal_global_cyc_outer][min_dist_path_ind]

    min_dist_path = min_dist_cyc_outer; min_dist_path_pts = points_global_cyc_outer; path_type = 'cyc_outer'
    tang_global_path = tang_global_cyc_outer; tang_normal_global_path = tang_normal_global_cyc_outer; surf_normal_global_path = surf_normal_global_cyc_outer
    
    return min_dist_path, min_dist_path_pts, tang_global_path, tang_normal_global_path, surf_normal_global_path, path_type