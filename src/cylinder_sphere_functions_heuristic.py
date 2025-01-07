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

# Including the following command to ensure that python is able to find the relevant files afer changing directory
sys.path.insert(0, '')
# Obtaining the current directory
cwd = os.getcwd()

# Importing code for plotting
rel_path = '\Cylinder code'
os.chdir(cwd + rel_path)
from plotting_class import plotting_functions
# Importing code for the cylinder
from Cylinder_2D_Dubins_functions_simple import generate_visualize_path

# Importing code for the sphere
rel_path = '\Sphere code'
os.chdir(cwd + rel_path)
from Path_generation_sphere import optimal_path_sphere_three_seg, generate_points_sphere

# Importing code for the plane
rel_path = '\Plane code'
os.chdir(cwd + rel_path)
from Plane_Dubins_functions import optimal_dubins_path

# Returning to initial directory
os.chdir(cwd)

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

def Path_generation_sphere_plane_sphere(ini_config, fin_config, center_ini_sphere, center_fin_sphere,\
                                           r, R, axis_plane, ht_plane, disc_no, plot_figure_configs,\
                                           visualization = 1, filename = "temp.html"):
    '''
    In this function, the paths connecting a given pair of spheres (inner or outer) with
    a cross-tangent plane is generated.

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
    center_ini_sphere : Array
        Contains the position of the center of the initial sphere.
    center_fin_sphere : Array
        Contains the position of the center of the final sphere.
    r: Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the surface.
    axis_plane : Array
        Axis of the line connecting the centers of the two spheres.
    ht_plane : Scalar
        Length of the plane connecting the considered pair of spheres.
    disc_no : Scalar
        Number of discretizations in theta and phi considered, where theta represents
        the angle on the base and top of the cylinder, and phi represents heading angle
        at the base and top of the cylinder.
    plot_figure_configs : Plotly figure handle
        Figure handle corresponding to the plotly figure generated, which is utilized and
        updated if visualization = 1.
    visualization : Scalar, optional
        Variable to decide whether to show the plot of the configurations and the
        surfaces. Default is equal to 1.
    filename : String, optional
        Name of the file in which the figure should be written. Used when visualization
        is set to 1. Default is "temp.html".

    Returns
    -------

    '''

    # Discretizing the initial and final angles.
    # NOTE: thetai are generated such that they are in the interval
    # [0, 2pi). Therefore, they cannot take the value of 2pi, as this will then
    # cause a redundancy.
    thetai = np.linspace(0, 2*math.pi, disc_no, endpoint = False)
    phii = np.linspace(-math.pi/2, math.pi/2, disc_no)
    phio = np.linspace(-math.pi/2, math.pi/2, disc_no)

    # We generate a random vector and orthonormalize it with respect to the axis
    # of the cylinder to obtain the x-axis using which theta is defined.
    flag = 0; counter = 0
    tol = 10**(-2) # tolerance for the dot product
    while flag == 0:

        # Generating a random vector
        temp = np.random.rand(3)

        # Orthonormalizing using Gram Schmidt
        if np.linalg.norm(-np.dot(temp, axis_plane)*axis_plane + temp) > tol:

            # In this case, we have obtained the desired x-axis
            x = (-np.dot(temp, axis_plane)*axis_plane + temp)\
                /np.linalg.norm(-np.dot(temp, axis_plane)*axis_plane + temp)
            
            flag = 1

        else:
            
            # We check if we have exceeded a threshold counter to ensure that we do not
            # go into an infinite loop
            if counter > 5:
                raise Exception('Going into an infinite loop for generating the random vector')
            
            # Incrementing the counter
            counter += 1

    # Plotting the configurations, spheres, and cylinders if visualization is 1.
    if visualization == 1:       
        
        # Making a copy of the figure containing the initial and final configurations
        plot_figure = copy.deepcopy(plot_figure_configs)

        # Plotting the spheres at the initial and final configurations
        plot_figure.surface_3D(generate_points_sphere(center_ini_sphere, R)[0],\
                               generate_points_sphere(center_ini_sphere, R)[1],\
                               generate_points_sphere(center_ini_sphere, R)[2], 'grey',\
                               'Initial sphere', 0.7)
        plot_figure.surface_3D(generate_points_sphere(center_fin_sphere, R)[0],\
                               generate_points_sphere(center_fin_sphere, R)[1],\
                               generate_points_sphere(center_fin_sphere, R)[2], 'grey',\
                               'Final sphere', 0.7)
        
        plot_figure.update_layout_3D('X (m)', 'Y (m)', 'Z (m)',\
                                    'Visualization of surfaces connecting initial and final configurations' +\
                                    ' via plane')
        
        # Writing the figure on the html file
        plot_figure.writing_fig_to_html(filename, 'a')

    # Defining empty arrays to hold the path length for each discretization set
    sp_1_path_lengths = np.empty((len(thetai), len(phii)))
    sp_1_path_lengths[:] = np.NaN
    sp_2_path_lengths = np.empty((len(thetai), len(phio)))
    sp_2_path_lengths[:] = np.NaN
    plane_path_lengths = np.empty((len(phii), len(phio))) # Note that for plane, it depends on only the delta phi.
    plane_path_lengths[:] = np.NaN

    # Final array to store path lengths for all configurations
    path_lengths = np.empty((len(thetai), len(phii), len(phio)))

    # First, we obtain the path length on the plane
    for i in range(len(phii)):
        for j in range(len(phio)):

            # We obtain the initial and final configuration on the plane
            ini_config_plane = np.array([0, 0, phii[i]])
            fin_config_plane = np.array([math.sqrt(ht_plane**2 - 4*R**2), 0, phio[j]])

            plane_path_lengths[i, j] = optimal_dubins_path(ini_config_plane, fin_config_plane, r, False)[0]

    # Now, we obtain the path lengths on the initial sphere
    alpha = math.acos(2*R/ht_plane)
    
    for i in range(len(thetai)):

        # Obtaining the configuration for the sphere
        # print('Center of ini sphere', center_ini_sphere, ' axis plane is ', axis_plane, 'x is ', x)
        loc_ini = center_ini_sphere + R*math.cos(alpha)*axis_plane + R*math.sin(alpha)*(math.cos(thetai[i])*x + math.sin(thetai[i])*(np.cross(axis_plane, x)))
        # Obtaining the location corresponding to the exit sphere
        loc_fin = center_fin_sphere - R*math.cos(alpha)*axis_plane + R*math.sin(alpha)*(math.cos(thetai[i] + math.pi)*x \
                + math.sin(thetai[i] + math.pi)*(np.cross(axis_plane, x)))

        # Obtaining the tangent vector
        t = (loc_fin - loc_ini)/np.linalg.norm(loc_fin - loc_ini)

        for j in range(len(phii)):
            
            T_ini = math.cos(phii[j])*t + math.sin(phii[j])*(np.cross((loc_ini - center_ini_sphere)/R, t))

            # Now, we construct the configuration for planning on the initial sphere
            # ini_sphere_ini_loc = np.array([ini_config[0, i] - center_ini_sphere[i] for i in range(3)])
            # ini_sphere_ini_tang = np.array([ini_config[1, i] for i in range(3)])
            # ini_sphere_ini_tang_norm = np.cross(ini_sphere_ini_loc, ini_sphere_ini_tang)/R
            # ini_sphere_ini_config = np.array([[ini_sphere_ini_loc[0], ini_sphere_ini_tang[0], ini_sphere_ini_tang_norm[0]],\
            #                                   [ini_sphere_ini_loc[1], ini_sphere_ini_tang[1], ini_sphere_ini_tang_norm[1]],\
            #                                   [ini_sphere_ini_loc[2], ini_sphere_ini_tang[2], ini_sphere_ini_tang_norm[2]]])
            
            ini_sphere_ini_config = config_sphere(ini_config[0, :], center_ini_sphere, ini_config[1, :])

            # Now, we construct the configuration for exit on the initial sphere
            # ini_sphere_fin_loc = loc_ini - center_ini_sphere
            # ini_sphere_fin_tang_norm = np.cross((loc_ini - center_ini_sphere)/R, T_ini)
            # ini_sphere_fin_config = np.array([[ini_sphere_fin_loc[0], T_ini[0], ini_sphere_fin_tang_norm[0]],\
            #                                   [ini_sphere_fin_loc[1], T_ini[1], ini_sphere_fin_tang_norm[1]],\
            #                                   [ini_sphere_fin_loc[2], T_ini[2], ini_sphere_fin_tang_norm[2]]])

            print('Location on sphere is ', loc_ini, ' and center is ', center_ini_sphere, '. Norm is ', np.linalg.norm(loc_ini - center_ini_sphere))
            ini_sphere_fin_config = config_sphere(loc_ini, center_ini_sphere, T_ini)
            
            filename_sp = "sp_1_thetai_" + str(i) + "_phii_" + str(j) + ".html"
            _, sp_1_path_lengths[i, j], _, _, _, _ =\
                            optimal_path_sphere_three_seg(ini_sphere_ini_config, ini_sphere_fin_config, r, R, 1, filename_sp)

        for j in range(len(phio)):
            
            T_ini = math.cos(phio[j])*t + math.sin(phio[j])*(np.cross((loc_ini - center_ini_sphere)/R, t))

            # Now, we construct the configuration for planning on the final sphere
            fin_sphere_ini_config = config_sphere(loc_fin, center_fin_sphere, T_ini)

            # Now, we construct the configuration for exit on the final sphere
            fin_sphere_fin_config = config_sphere(fin_config[0, :], center_fin_sphere, fin_config[1, :])

            filename_sp = "sp_2_thetai_" + str(i) + "_phio_" + str(j) + ".html"
            _, sp_2_path_lengths[i, j], _, _, _, _ =\
                            optimal_path_sphere_three_seg(fin_sphere_ini_config, fin_sphere_fin_config, r, R, 1, filename_sp)

    min_dist = np.inf
    thetai_min = np.nan; phii_min = np.nan; phio_min = np.nan
    for i in range(len(thetai)):
        for j in range(len(phii)):
            for k in range(len(phio)):

                path_lengths[i, j, k] = sp_1_path_lengths[i, j] + plane_path_lengths[j, k] + sp_2_path_lengths[i, k]
                if path_lengths[i, j, k] < min_dist:

                    thetai_min = thetai[i]; phii_min = phii[j]; phio_min = phio[k]


    # We plot the optimal path
    # Obtaining the configuration for the spheres for exit from the first sphere and entry at final sphere
    loc_ini = center_ini_sphere + R*math.cos(alpha)*axis_plane +\
         R*math.sin(alpha)*(math.cos(thetai_min)*x + math.sin(thetai_min)*(np.cross(axis_plane, x)))
    # Obtaining the location corresponding to the exit sphere
    loc_fin = center_fin_sphere - R*math.cos(alpha)*axis_plane + R*math.sin(alpha)*(math.cos(thetai_min + math.pi)*x \
            + math.sin(thetai_min + math.pi)*(np.cross(axis_plane, x)))
    t = (loc_fin - loc_ini)/np.linalg.norm(loc_fin - loc_ini)
    # Obtaining the tangent vectors for the two spheres
    T_ini = math.cos(phii_min)*t + math.sin(phii_min)*(np.cross((loc_ini - center_ini_sphere)/R, t))
    T_fin = math.cos(phio_min)*t + math.sin(phio_min)*(np.cross((loc_ini - center_ini_sphere)/R, t))

    # Obtaining the optimal path on the first sphere
    ini_config_sphere = config_sphere(ini_config[0, :], center_ini_sphere, ini_config[1, :])
    fin_config_sphere = config_sphere(loc_ini, center_ini_sphere, T_ini)
    # Obtaining the best feasible path's portion on the first sphere
    _, _, _, minlen_sp1_path_points_x, minlen_sp1_path_points_y, minlen_sp1_path_points_z =\
        optimal_path_sphere_three_seg(ini_config_sphere, fin_config_sphere, r, R, 1, "sp1_optimal_cross_tangent.html")
    
    # Obtaining the optimal path on the final sphere
    ini_config_sphere = config_sphere(loc_fin, center_fin_sphere, T_fin)
    fin_config_sphere = config_sphere(fin_config[0, :], center_fin_sphere, fin_config[1, :])
    # Obtaining the best feasible path's portion on the second sphere
    _, _, _, minlen_sp2_path_points_x, minlen_sp2_path_points_y, minlen_sp2_path_points_z =\
        optimal_path_sphere_three_seg(ini_config_sphere, fin_config_sphere, r, R, 1, "sp2_optimal_cross_tangent.html")

    # Obtaining the optimal path on the plane
    ini_config_plane = np.array([0, 0, phii_min])
    fin_config_plane = np.array([math.sqrt(ht_plane**2 - 4*R**2), 0, phio_min])
    _, _, _, pts_x, pts_y = optimal_dubins_path(ini_config_plane, fin_config_plane, r, 'optimal_path_cross_tangent_plane.html')

    # Finding the global points of the path on the first sphere using a coordinate transformation
    points_global_sp1 = np.empty((len(minlen_sp1_path_points_x), 3))
    for i in range(len(minlen_sp1_path_points_x)):

        points_global_sp1[i, 0] = minlen_sp1_path_points_x[i] + center_ini_sphere[0]
        points_global_sp1[i, 1] = minlen_sp1_path_points_y[i] + center_ini_sphere[1]
        points_global_sp1[i, 2] = minlen_sp1_path_points_z[i] + center_ini_sphere[2]

    # Finding the global points of the path on the last sphere using a coordinate transformation
    points_global_sp2 = np.empty((len(minlen_sp2_path_points_x), 3))
    for i in range(len(minlen_sp2_path_points_x)):

        points_global_sp2[i, 0] = minlen_sp2_path_points_x[i] + center_fin_sphere[0]
        points_global_sp2[i, 1] = minlen_sp2_path_points_y[i] + center_fin_sphere[1]
        points_global_sp2[i, 2] = minlen_sp2_path_points_z[i] + center_fin_sphere[2]

    # Finding the global points of the path on the cross-tangent plane
    points_global_plane = np.empty((len(pts_x), 3))
    # We construct the rotation matrix relating the frame corresponding to the plane.
    rot_mat = np.array([[t[0], np.cross((loc_ini - center_ini_sphere)/R, t)[0], ((loc_ini - center_ini_sphere)/R)[0]],\
                        [t[1], np.cross((loc_ini - center_ini_sphere)/R, t)[1], ((loc_ini - center_ini_sphere)/R)[1]],\
                        [t[2], np.cross((loc_ini - center_ini_sphere)/R, t)[2], ((loc_ini - center_ini_sphere)/R)[2]]])
    
    for i in range(len(points_global_plane)):

        pos = loc_ini + np.matmul(rot_mat, np.array([pts_x[i], pts_y[i], 0]))
        points_global_plane[i, 0] = pos[0]
        points_global_plane[i, 1] = pos[1]
        points_global_plane[i, 2] = pos[2]

    print('Including near-optimal path')
    # Plotting the path on the first sphere
    plot_figure.scatter_3D(points_global_sp1[:, 0], points_global_sp1[:, 1],\
                            points_global_sp1[:, 2], 'blue', 'Optimal path')  
    # Plotting the path on the cylinder
    plot_figure.scatter_3D(points_global_plane[:, 0], points_global_plane[:, 1],\
                            points_global_plane[:, 2], 'blue', False)
    # Updating the figure with path on the last sphere
    plot_figure.scatter_3D(points_global_sp2[:, 0], points_global_sp2[:, 1],\
                            points_global_sp2[:, 2], 'blue', False)
                    
    # Writing the figure on the html file
    plot_figure.writing_fig_to_html(filename, 'a')

def Path_generation_sphere_sphere_sphere(ini_config, fin_config, center_ini_sphere, center_fin_sphere,\
                                           r, R, axis_plane, ht_plane, disc_no, plot_figure_configs,\
                                           visualization = 1, filename = "temp.html"):
    '''
    In this function, the paths connecting a given pair of spheres (inner or outer) with
    an interemediary sphere is generated.

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
    center_ini_sphere : Array
        Contains the position of the center of the initial sphere.
    center_fin_sphere : Array
        Contains the position of the center of the final sphere.
    r: Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the surface.
    axis_plane : Array
        Axis of the line connecting the centers of the two spheres.
    ht_plane : Scalar
        Length of the plane connecting the considered pair of spheres.
    disc_no : Scalar
        Number of discretizations in the parameters considered, which correspond
        to the angle corresponding to parameterizing the center of the intermediary sphere,
        the angle corresponding to the tangent vector corresponding to exit from the initial
        sphere, and the angle corresponding to the tangent vector for entry onto the final sphere.
    plot_figure_configs : Plotly figure handle
        Figure handle corresponding to the plotly figure generated, which is utilized and
        updated if visualization = 1.
    visualization : Scalar, optional
        Variable to decide whether to show the plot of the configurations and the
        surfaces. Default is equal to 1.
    filename : String, optional
        Name of the file in which the figure should be written. Used when visualization
        is set to 1. Default is "temp.html".

    Returns
    -------

    '''

    # Discretizing the angle for parameterizing the intermediary sphere and the
    # angles for the tangent vector for exit from initial sphere and entry into final
    # sphere
    theta

def config_sphere(loc_3d, loc_center_sphere, tang_sphere):
    '''
    In this function, a rotation matrix is constructed for the sphere considering
    the global frame to be shifted to the center of the sphere.
    
    Parameters
    ----------
    loc_3d : Array
        Contains the location in 3D in the global frame.
    loc_center_sphere : Array
        Contains the location in 3D of the center of the considered sphere in the
        global frame.
    tang_sphere : Array
        Contains the direction cosines of the unit vector corresponding to the tangent
        vector represented in the global frame.
    
    Returns
    -------
    R : 3x3 matrix
        Contains the rotation matrix corresponding to the configuration on the sphere.

    '''

    # Computing the location with respect to the center of the sphere
    loc_sphere = loc_3d - loc_center_sphere
    # Compute the tangent-normal
    tang_norm = np.cross(loc_sphere, tang_sphere)/np.linalg.norm(np.cross(loc_sphere, tang_sphere))
    R = np.array([[loc_sphere[0], tang_sphere[0], tang_norm[0]],\
                  [loc_sphere[1], tang_sphere[1], tang_norm[1]],\
                  [loc_sphere[2], tang_sphere[2], tang_norm[2]]])
    
    return R

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
    
    # Computing the center of the left and right spheres at the initial and final
    # configurations : ASSUMING THAT THE LEFT SPHERE LIES ON THE POSITIVE SIDE OF
    # THE SURFACE NORMAL VECTOR, i.e., SURFACE NORMAL IS OPPOSITE TO THE RADIAL VECTOR
    # FOR THE LEFT SPHERE
    
    ini_loc_left_sp = ini_config[0, :] + R*ini_config[3, :]
    ini_loc_right_sp = ini_config[0, :] - R*ini_config[3, :]
    fin_loc_left_sp = fin_config[0, :] + R*fin_config[3, :]
    fin_loc_right_sp = fin_config[0, :] - R*fin_config[3, :]

    # Computing the axis of each cylinder connecting the pair of inner and outer spheres
    # at the initial and final configurations.
    # Obtaining the unit vector along the axis of the inner sphere connections    
    axis_ii = (fin_loc_left_sp - ini_loc_left_sp)\
        /np.linalg.norm(fin_loc_left_sp - ini_loc_left_sp)
    ht_ii = np.linalg.norm(fin_loc_left_sp - ini_loc_left_sp)
    # Obtaining the unit vector along the axis of the outer sphere connections
    axis_oo = (fin_loc_right_sp - ini_loc_right_sp)\
        /np.linalg.norm(fin_loc_right_sp - ini_loc_right_sp)
    ht_oo = np.linalg.norm(fin_loc_right_sp - ini_loc_right_sp)

    # Obtaining the line segment corresponding to the axis of the cross-tangent planes
    axis_io = (fin_loc_right_sp - ini_loc_left_sp)/np.linalg.norm(fin_loc_right_sp - ini_loc_left_sp)
    ht_io = np.linalg.norm(fin_loc_right_sp - ini_loc_left_sp)
    axis_oi = (fin_loc_left_sp - ini_loc_right_sp)/np.linalg.norm(fin_loc_left_sp - ini_loc_right_sp)
    ht_oi = np.linalg.norm(fin_loc_left_sp - ini_loc_right_sp)
    
    if visualization == 1:

        # Creating a plotly figure environment
        plot_figure_configs = plotting_functions()

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
        
    # # We now construct the best feasible path connecting the two inner spheres
    # Path_generation_sphere_cylinder_sphere(ini_config, fin_config, ini_loc_left_sp, fin_loc_left_sp,\
    #                                        r, R, axis_ii, ht_ii, disc_no, plot_figure_configs, 1, filename)
    
    # We do the same task for cross-tangent connections
    if visualization == 1:

        # Creating a plotly figure environment
        plot_figure_configs = plotting_functions()

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
        
        plot_figure_configs.update_layout_3D('X (m)', 'Y (m)', 'Z (m)', 'Visualization of configurations for cross-tangent')
            
        # Writing the figure on the html file
        plot_figure_configs.writing_fig_to_html(filename, 'a')

    # We perform the connections through three sphere setup as well.

        
    # We now construct the best feasible path connecting the two cross-tangent spheres
    Path_generation_sphere_plane_sphere(ini_config, fin_config, ini_loc_left_sp, fin_loc_right_sp,\
                                           r, R, axis_io, ht_io, disc_no, plot_figure_configs, 1, filename)


def generate_points_cylinder(start_pt, axis, R, height_axis, x_axis):
    '''
    This function generates points on the cylinder whose origin is at start_pt
    and whose axis is along the variable "axis".

    Parameters
    ----------
    start_pt : Numpy 1x3 array
        Contains the coordinates of the origin of the cylinder in the global frame.
    axis : Numpy 1x3 array
        Contains the direction cosines of the axis of the cylinder.
    R : Scalar
        Radius of the cylinder.
    height_axis : Scalar
        Describes the height of the cylinder (along the axis of the cylinder).

    Returns
    -------
    x_grid_global : Numpy nd array
        Contains the x-coordinate of the points generated on the cylinder.
    y_grid_global : Numpy nd array
        Contains the y-coordinate of the points generated on the cylinder.
    z_grid_global : Numpy nd array
        Contains the z-coordinate of the points generated on the cylinder.
    '''
    
    z = np.linspace(0, height_axis, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    
    # Finding the coordinates of the points on the cylinder in the body frame    
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = R*np.cos(theta_grid)
    y_grid = R*np.sin(theta_grid)
   
    # Finding the direction cosines of the y-axis
    y_axis = np.cross(axis, x_axis)
    
    # Defining the rotation matrix
    rot_mat = np.array([[x_axis[0], y_axis[0], axis[0]],\
                        [x_axis[1], y_axis[1], axis[1]],\
                        [x_axis[2], y_axis[2], axis[2]]])
        
    # Finding the coordinates of the points on the cylinder in the global frame using
    # a translation to the origin of the cylinder (given by the variable start_pt) and
    # a rotation of the axes of the frame
    x_grid_global = np.empty(np.shape(x_grid))
    y_grid_global = np.empty(np.shape(x_grid))
    z_grid_global = np.empty(np.shape(x_grid))
    
    for i in range(np.shape(x_grid)[0]):
        for j in range(np.shape(x_grid)[1]):
            
            # Coordinates in the global frame
            r_global = start_pt + np.matmul(rot_mat, np.array([x_grid[i, j], y_grid[i, j], z_grid[i, j]]))
            x_grid_global[i, j] = r_global[0]
            y_grid_global[i, j] = r_global[1]
            z_grid_global[i, j] = r_global[2]
        
    return x_grid_global, y_grid_global, z_grid_global