# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 20:51:49 2022

@author: deepa
"""

import numpy as np
import os
import math
import copy

# Importing code for plotting
path = 'D:\TAMU\Research\Cylinder code'
os.chdir(path)
from plotting_class import plotting_functions

# Importing code for the sphere
path = 'D:\TAMU\Research\Athindra files\Codes'
os.chdir(path)
from IconfigFconfig import CustomInputFromOtherModule
from Output import PP_modified

# Importing code for the cylinder
path = 'D:\TAMU\Research\Cylinder code'
os.chdir(path)
# from Cylinder_2D_Dubins_functions import generate_visualize_path_simple
from Cylinder_2D_Dubins_functions_simple import generate_visualize_path

# Returning to initial directory
path = 'D:\TAMU\Research\Cylinder, sphere, and cone'

os.chdir(path)

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
    
    # Computing the center of the left and right spheres at the initial and final
    # configurations : ASSUMING THAT THE LEFT SPHERE LIES ON THE POSITIVE SIDE OF
    # THE SURFACE NORMAL VECTOR, i.e., SURFACE NORMAL IS OPPOSITE TO THE RADIAL VECTOR
    # FOR THE LEFT SPHERE
    
    ini_config_left_sp = ini_config[0, :] + R*ini_config[3, :]
    ini_config_right_sp = ini_config[0, :] - R*ini_config[3, :]
    fin_config_left_sp = fin_config[0, :] + R*fin_config[3, :]
    fin_config_right_sp = fin_config[0, :] - R*fin_config[3, :]
    
    # Computing the axis of each cylinder connecting one of the initial and final
    # spheres. Nomenclature followed: axis_ij represents axis of the cylinder connecting
    # the ith initial sphere and the jth final sphere (i, j = {l, r})
    
    axis_ll = fin_config_left_sp - ini_config_left_sp
    ht_cyl_ll = np.linalg.norm(axis_ll)
    axis_ll = axis_ll/np.linalg.norm(axis_ll) # modifying it to be a unit vector
    axis_lr = fin_config_right_sp - ini_config_left_sp
    ht_cyl_lr = np.linalg.norm(axis_lr)
    axis_lr = axis_lr/np.linalg.norm(axis_lr) # modifying it to be a unit vector
    axis_rl = fin_config_left_sp - ini_config_right_sp
    ht_cyl_rl = np.linalg.norm(axis_rl)
    axis_rl = axis_rl/np.linalg.norm(axis_rl) # modifying it to be a unit vector
    axis_rr = fin_config_right_sp - ini_config_right_sp
    ht_cyl_rr = np.linalg.norm(axis_rr)
    axis_rr = axis_rr/np.linalg.norm(axis_rr) # modifying it to be a unit vector
    
    # Generating a random vector and orthonormalizing it wrt axis of cylinder - 
    # DOING THIS FOR LL and RR ONLY.
    # This task is to generate the x-axis for the cylinders
    temp = np.random.rand(3)
    # Orthonormalizing using Gram Schmidt
    tol = 10**(-2) # tolerance for the dot product
    # Checking if the vector generated is along the axis of the cylinder (which is
    # the z-axis for the respective cylinder) or not.
    # if np.linalg.norm(-np.dot(temp, axis_rr)*axis_rr + temp) < tol:
        
    #     raise Exception('Regenerate the random vectors.')
        
    # else:
        
    #     xaxis_rr = (temp - np.dot(temp, axis_rr)*axis_rr)/np.linalg.norm(temp - np.dot(temp, axis_rr)*axis_rr)
    if np.linalg.norm(-np.dot(temp, axis_ll)*axis_ll + temp) < tol or\
        np.linalg.norm(-np.dot(temp, axis_rr)*axis_rr + temp) < tol:
        
        raise Exception('Regenerate the random vector.')
        
    else:
        
        xaxis_ll = (temp - np.dot(temp, axis_ll)*axis_ll)/np.linalg.norm(temp - np.dot(temp, axis_ll)*axis_ll)
        xaxis_rr = (temp - np.dot(temp, axis_rr)*axis_rr)/np.linalg.norm(temp - np.dot(temp, axis_rr)*axis_rr)
    
    # Discretizing the initial and final angles - DOING THIS FOR RR ONLY.
    # NOTE: thetai and thetao are generated such that they are in the interval
    # [0, 2pi). Therefore, they cannot take the value of 2pi, as this will then
    # cause a redundancy.
    thetai = np.linspace(0, 2*math.pi, disc_no, endpoint = False)
    thetao = np.linspace(0, 2*math.pi, disc_no, endpoint = False)
    phii = np.linspace(0, math.pi, disc_no)
    phio = np.linspace(0, math.pi, disc_no)
    
    if visualization == 1:
        
        # Generating arrays for ll and rr connections of the spheres. This will
        # be used in the for loop below to generate paths on the ll and rr setups.
        
        # Array for connection type
        conn_type = np.array(['left-left', 'right-right'])
        # Array for sphere locations
        ini_config_spheres = np.array([ini_config_left_sp, ini_config_right_sp])
        fin_config_spheres = np.array([fin_config_left_sp, fin_config_right_sp])
        # Array for axis and height of the cylinders
        axis_cylinders = np.array([axis_ll, axis_rr])
        xaxis_cylinders = np.array([xaxis_ll, xaxis_rr])
        ht_cylinders = np.array([ht_cyl_ll, ht_cyl_rr])
        
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
        
        # Making a copy of the figure containing the initial and final configurations
        plot_figure = copy.deepcopy(plot_figure_configs)
            
        # Visualizing the connections via both left-left spheres and right-right spheres
        for m in range(len(conn_type)):
            
            # Plotting the spheres at the initial and final configurations
            plot_figure.surface_3D(generate_points_sphere(ini_config_spheres[m], R)[0],\
                                   generate_points_sphere(ini_config_spheres[m], R)[1],\
                                   generate_points_sphere(ini_config_spheres[m], R)[2], 'grey',\
                                   'Initial sphere', 0.7)
            plot_figure.surface_3D(generate_points_sphere(fin_config_spheres[m], R)[0],\
                                   generate_points_sphere(fin_config_spheres[m], R)[1],\
                                   generate_points_sphere(fin_config_spheres[m], R)[2], 'grey',\
                                   'Final sphere', 0.7)
        
            # Plotting the cylinders connecting the spheres
            x_cyl_conn_type, y_cyl_conn_type, z_cyl_conn_type =\
                generate_points_cylinder(ini_config_spheres[m], axis_cylinders[m], R, ht_cylinders[m])
            plot_figure.surface_3D(x_cyl_conn_type, y_cyl_conn_type, z_cyl_conn_type, 'lightgreen', 'Cylinder', 0.4)        
                
            plot_figure.update_layout_3D('X (m)', 'Y (m)', 'Z (m)',\
                                         'Visualization of surfaces connecting initial and final configurations' +\
                                         ' via both left-left and right-right spheres')
            
        # Writing the figure on the html file
        plot_figure.writing_fig_to_html(filename, 'w')
        
        # Generating the paths along the left-left and right-right spheres
        for m in range(len(conn_type)):
            
            # Making a copy of the figure containing the initial and final configurations
            plot_figure = copy.deepcopy(plot_figure_configs)
            
            # Plotting the spheres at the initial and final configurations
            plot_figure.surface_3D(generate_points_sphere(ini_config_spheres[m], R)[0],\
                                   generate_points_sphere(ini_config_spheres[m], R)[1],\
                                   generate_points_sphere(ini_config_spheres[m], R)[2], 'grey',\
                                   'Initial sphere', 0.7)
            plot_figure.surface_3D(generate_points_sphere(fin_config_spheres[m], R)[0],\
                                   generate_points_sphere(fin_config_spheres[m], R)[1],\
                                   generate_points_sphere(fin_config_spheres[m], R)[2], 'grey',\
                                   'Final sphere', 0.7)
        
            # Plotting the cylinders connecting the spheres
            x_cyl_conn_type, y_cyl_conn_type, z_cyl_conn_type =\
                generate_points_cylinder(ini_config_spheres[m], axis_cylinders[m], R, ht_cylinders[m])
            plot_figure.surface_3D(x_cyl_conn_type, y_cyl_conn_type, z_cyl_conn_type, 'lightgreen', 'Cylinder', 0.4)        
                
            plot_figure.update_layout_3D('X (m)', 'Y (m)', 'Z (m)',\
                                         'Visualization of surfaces connecting initial and final configurations' +\
                                         ' via ' + conn_type[m] + ' spheres')
                
            # Writing the figure on the html file
            plot_figure.writing_fig_to_html(filename, 'a')
        
            # Plotting the xaxis for the cylinder
            plot_figure.arrows_3D([ini_config_spheres[m, 0]], [ini_config_spheres[m, 1]], [ini_config_spheres[m, 2]],\
                                  [xaxis_cylinders[m, 0]], [xaxis_cylinders[m, 1]], [xaxis_cylinders[m, 2]], 'grey',\
                                  'greys', False, 5, 5, 4, 'n')
            plot_figure.arrows_3D([fin_config_right_sp[0]], [fin_config_right_sp[1]], [fin_config_right_sp[2]],\
                                  [xaxis_cylinders[m, 0]], [xaxis_cylinders[m, 1]], [xaxis_cylinders[m, 2]], 'grey',\
                                  'greys', False, 5, 5, 4, 'n')
        
            # Creating arrays for storing the paths for the various configurations,
            # and initializing all variables to NaN
            sp_1_rr_path_lengths = np.empty((len(thetai), len(phii)))
            sp_1_rr_path_lengths[:] = np.NaN
            sp_2_rr_path_lengths = np.empty((len(thetao), len(phio)))
            sp_2_rr_path_lengths[:] = np.NaN
            if len(thetai) != len(thetao):
                
                raise Exception("Unequal number of discretizations for thetai and thetao")
                
            else:
                
                cyl_rr_path_lengths = np.empty((len(phii), len(thetao), len(phio)))
                cyl_rr_path_lengths[:] = np.NaN
            
            # Final array to store path lengths for all configurations        
            path_lengths = np.empty((len(thetai), len(phii), len(thetao), len(phio)))
            
            # Storing the configuration corresponding to the minimum path length
            min_dist = np.infty
            thetai_min = 0
            phii_min = 0
            thetao_min = 0
            phio_min = 0
        
        # Computing path lengths for all the configurations and selecting the
        # optimal configuration
        for i in range(len(thetai)):
            for j in range(len(phii)):
                for k in range(len(thetao)):
                    for l in range(len(phio)):
                        
                        # Obtaining the configuration of the Dubins vehicle
                        # at point of entry and exit from the cylinder
                        Tic, Toc, Pic, Poc, TicB, TocB, PicB, PocB, _ = \
                            configurations_discrete_angles(thetai[i], phii[j], thetao[k], phio[l], ini_config_right_sp,\
                                                            xaxis_rr, axis_rr, ht_cyl_rr, R)
                            
                        # Finding the minimum distance on the first sphere
                        # Position of configurations on the sphere wrt origin of frame placed
                        # at the center of the initial sphere is passed     
                        if np.isnan(sp_1_rr_path_lengths[i, j]): # Checking if path length was already computed
                            
                            filename_sp = "sp_1_rr_thetai_" + str(i) + "_phii_" + str(j) + ".html"
                            sp_1_rr_path_lengths[i, j], _, _, _, _, _, _ = \
                                CustomInputFromOtherModule(filename_sp, 'p', ini_config[0, :] - ini_config_right_sp,\
                                                            Pic - ini_config_right_sp, ini_config[1, :], Tic, r, R, 10)
                        
                        # Finding the minimum distance on the cylinder
                        # For this purpose, the configurations in the frame fixed
                        # at the origin of the cylinder is passed
                        if np.isnan(cyl_rr_path_lengths[j, k, l]): # Checking if path length was already computed
                        
                            filename_cyc = "cyc_rr_phii_" + str(j) + "_thetao_" + str(k) + "_phio_" + str(l) + ".html"
                            # cyl_rr_path_lengths[j, k, l], _, _ = generate_visualize_path_simple(PicB, TicB, R, PocB, TocB,\
                            #            ht_cyl_rr, 2, r, filename_cyc)
                            cyl_rr_path_lengths[j, k, l], _, _ = generate_visualize_path(PicB, TicB, R, PocB, TocB,\
                                        ht_cyl_rr, 1, r, filename_cyc)
                        
                        # Finding the minimum distance on the last sphere
                        # Position of configurations on the sphere wrt origin of frame placed
                        # at the center of the initial sphere is passed
                        if np.isnan(sp_2_rr_path_lengths[k, l]): # Checking if path length was already computed
                            
                            filename_sp = "sp_2_rr_thetao_" + str(k) + "_phio_" + str(l) + ".html"
                            sp_2_rr_path_lengths[k, l], _, _, _, _, _, _ = \
                                CustomInputFromOtherModule(filename_sp, 'p', Poc - fin_config_right_sp,\
                                                            fin_config[0, :] - fin_config_right_sp, Toc, fin_config[1, :],\
                                                            r, R, 10)                       
        
                        path_lengths[i, j, k, l] = sp_1_rr_path_lengths[i, j] + \
                            cyl_rr_path_lengths[j, k, l] + sp_2_rr_path_lengths[k, l]
                            
                        # Checking if obtained solution is better
                        if path_lengths[i, j, k, l] < min_dist:
                            
                            min_dist = path_lengths[i, j, k, l]
                            thetai_min = thetai[i]
                            phii_min = phii[j]
                            thetao_min = thetao[k]
                            phio_min = phio[l]
        
        # Obtaining the configuration of the Dubins vehicle
        # at point of entry and exit from the cylinder corresponding to the
        # best discretization
        Tic, Toc, Pic, Poc, TicB, TocB, PicB, PocB, R_comp = \
            configurations_discrete_angles(thetai_min, phii_min, thetao_min, phio_min, ini_config_right_sp,\
                                            xaxis_rr, axis_rr, ht_cyl_rr, R)
        
        # Obtaining the variables for plotting
        # Obtaining variables for plotting on the first sphere
        filename_sp = "sp_1_rr_optimal.html"
        minlen_sp1, mintype_sp1, minFig_sp1, _, _, _, _ =\
            CustomInputFromOtherModule(filename_sp, 'p', ini_config[0, :] - ini_config_right_sp,\
                                        Pic - ini_config_right_sp, ini_config[1, :], Tic, r, R, 50)
        # Obtaining variables for plotting on the cylinder
        filename_cyc = "cyc_rr_optimal.html"
        # minlen_cyc, mintype_cyc, points_body_cyc = generate_visualize_path_simple(PicB, TicB, R, PocB, TocB,\
        #                                                                  ht_cyl_rr, 2, r, filename_cyc)
        minlen_cyc, mintype_cyc, points_body_cyc = generate_visualize_path(PicB, TicB, R, PocB, TocB,\
                                                                          ht_cyl_rr, 1, r, filename_cyc)
        # Obtaining variables for plotting on the last sphere
        filename_sp = "sp_2_rr_optimal.html"
        minlen_sp2, mintype_sp2, minFig_sp2, _, _, _, _ =\
            CustomInputFromOtherModule(filename_sp, 'p', Poc - fin_config_right_sp,\
                                        fin_config[0, :] - fin_config_right_sp, Toc, fin_config[1, :],\
                                        r, R, 50)
                
        # Finding the global points of the path on the cylinder using a coordinate
        # transformation
        points_global_cyc = np.empty(np.shape(points_body_cyc))
        for i in range(np.shape(points_body_cyc)[0]): # iterating through all points which are present row-wise
        
            points_global_cyc[i, :] = np.matmul(R_comp, points_body_cyc[i, :]) + ini_config_right_sp
        
        # Adding the plots for the optimal configuration
        plot_figure.arrows_3D([Pic[0]], [Pic[1]], [Pic[2]], [Tic[0]], [Tic[1]], [Tic[2]],\
                              'brown', 'brwnyl', False)
        plot_figure.arrows_3D([Poc[0]], [Poc[1]], [Poc[2]], [Toc[0]], [Toc[1]], [Toc[2]],\
                              'brown', 'brwnyl', False)        
        
        # Updating the figure with path on the first sphere
        PP_modified(plot_figure, 'Optimal path', [0], minFig_sp1[2], minFig_sp1[3], [0], ini_config_right_sp)        
        # Plotting the path on the cylinder
        plot_figure.scatter_3D(points_global_cyc[:, 0], points_global_cyc[:, 1],\
                                points_global_cyc[:, 2], 'blue', False)
        # Updating the figure with path on the first sphere
        PP_modified(plot_figure, 'Optimal path', [0], minFig_sp2[2], minFig_sp2[3], [0], fin_config_right_sp)
                        
        # Writing the figure on the html file
        plot_figure.writing_fig_to_html(filename, 'a')
        
        # Writing onto the file
        str_write = ["\n"]
        str_write.append('Total length of the feasible path is ' + str(min_dist) + '.')
        str_write.append('Angles corresponding to position of entry and exit from the cylinder are ' + str(thetai_min*180/math.pi)\
                          + ' and ' + str(thetao_min*180/math.pi) + ' degrees, respectively.')
        str_write.append('Angles corresponding to headings at entry and exit from the cylinder are ' + str(phii_min*180/math.pi)\
                          + ' and ' + str(phio_min*180/math.pi) + ' degrees, respectively.')
        str_write.append('Feasible path has a length of ' + str(minlen_sp1) + ' on the first sphere, ' +\
                      str(minlen_cyc) + ' on the cylinder, and ' + str(minlen_sp2) + ' on the final sphere.')
        str_write.append('Feasible path type on first sphere is ' + mintype_sp1 + '.')
        str_write.append('Feasible path type on cylinder is ' + mintype_cyc.upper() + '.')
        str_write.append('Feasible path type on the final sphere is ' + mintype_sp2 + '.')
        with open(filename, 'a') as f:
            for i in range(len(str_write)):
                
                if(str_write[i] == "\n"):
                    
                    f.write("<br />")
                
                else:
                    
                    f.write(str_write[i] + "<br />")
        
        # OLD CODE FOR JUST RR CONNECTION TYPE
        
        # # Creating a plotly figure environment
        # plot_figure = plotting_functions()
        
        # # Showing the initial and final points
        # plot_figure.points_3D([ini_config[0, 0]], [ini_config[0, 1]], [ini_config[0, 2]],\
        #                       'red', 'Initial point', 'circle')
        # plot_figure.points_3D([fin_config[0, 0]], [fin_config[0, 1]], [fin_config[0, 2]],\
        #                       'black', 'Final point', 'diamond')
        # # Showing the initial and final orientations
        # plot_figure.arrows_3D([ini_config[0, 0]], [ini_config[0, 1]], [ini_config[0, 2]],\
        #                       [ini_config[1, 0]], [ini_config[1, 1]], [ini_config[1, 2]],\
        #                       'orange', 'oranges', 'Tangent vector', 5, 5, 4, 'n')
        # plot_figure.arrows_3D([ini_config[0, 0]], [ini_config[0, 1]], [ini_config[0, 2]],\
        #                       [ini_config[2, 0]], [ini_config[2, 1]], [ini_config[2, 2]],\
        #                       'purple', 'purp', 'Tangent normal vector', 5, 5, 4, 'n')
        # plot_figure.arrows_3D([ini_config[0, 0]], [ini_config[0, 1]], [ini_config[0, 2]],\
        #                       [ini_config[3, 0]], [ini_config[3, 1]], [ini_config[3, 2]],\
        #                       'green', 'greens', 'Surface normal vector', 5, 5, 4, 'n')
        # plot_figure.arrows_3D([fin_config[0, 0]], [fin_config[0, 1]], [fin_config[0, 2]],\
        #                       [fin_config[1, 0]], [fin_config[1, 1]], [fin_config[1, 2]],\
        #                       'orange', 'oranges', False, 5, 5, 4, 'n')
        # plot_figure.arrows_3D([fin_config[0, 0]], [fin_config[0, 1]], [fin_config[0, 2]],\
        #                       [fin_config[2, 0]], [fin_config[2, 1]], [fin_config[2, 2]],\
        #                       'purple', 'purp', False, 5, 5, 4, 'n')
        # plot_figure.arrows_3D([fin_config[0, 0]], [fin_config[0, 1]], [fin_config[0, 2]],\
        #                       [fin_config[3, 0]], [fin_config[3, 1]], [fin_config[3, 2]],\
        #                       'green', 'greens', False, 5, 5, 4, 'n')
            
        # # Plotting the spheres at the initial and final configurations
        # plot_figure.surface_3D(generate_points_sphere(ini_config_right_sp, R)[0],\
        #                        generate_points_sphere(ini_config_right_sp, R)[1],\
        #                        generate_points_sphere(ini_config_right_sp, R)[2], 'grey',\
        #                        'Initial sphere', 0.7)
        # plot_figure.surface_3D(generate_points_sphere(fin_config_right_sp, R)[0],\
        #                        generate_points_sphere(fin_config_right_sp, R)[1],\
        #                        generate_points_sphere(fin_config_right_sp, R)[2], 'grey',\
        #                        'Final sphere', 0.7)
        
        # # Plotting the cylinders connecting the spheres
        # x_cyl_rr, y_cyl_rr, z_cyl_rr = generate_points_cylinder(ini_config_right_sp, axis_rr, R, ht_cyl_rr)
        # plot_figure.surface_3D(x_cyl_rr, y_cyl_rr, z_cyl_rr, 'lightgreen', 'Cylinder 1', 0.4)        
            
        # plot_figure.update_layout_3D('X (m)', 'Y (m)', 'Z (m)',\
        #                              'Visualization of surfaces connecting initial and final configurations')
            
        # # Writing the figure on the html file
        # plot_figure.writing_fig_to_html(filename, 'w')
        
        # # Plotting the xaxis for the cylinder
        # plot_figure.arrows_3D([ini_config_right_sp[0]], [ini_config_right_sp[1]], [ini_config_right_sp[2]],\
        #                       [xaxis_rr[0]], [xaxis_rr[1]], [xaxis_rr[2]], 'grey', 'greys', False, 5, 5, 4, 'n')
        # plot_figure.arrows_3D([fin_config_right_sp[0]], [fin_config_right_sp[1]], [fin_config_right_sp[2]],\
        #                       [xaxis_rr[0]], [xaxis_rr[1]], [xaxis_rr[2]], 'grey', 'greys', False, 5, 5, 4, 'n')
        
        # # Computing the paths on the RR cylinder
        # # Creating arrays for storing the paths for the various configurations,
        # # and initializing all variables to NaN
        # sp_1_rr_path_lengths = np.empty((len(thetai), len(phii)))
        # sp_1_rr_path_lengths[:] = np.NaN
        # sp_2_rr_path_lengths = np.empty((len(thetao), len(phio)))
        # sp_2_rr_path_lengths[:] = np.NaN
        # if len(thetai) != len(thetao):
            
        #     raise Exception("Unequal number of discretizations for thetai and thetao")
            
        # else:
            
        #     cyl_rr_path_lengths = np.empty((len(phii), len(thetao), len(phio)))
        #     cyl_rr_path_lengths[:] = np.NaN
        
        # # Final array to store path lengths for all configurations        
        # path_lengths = np.empty((len(thetai), len(phii), len(thetao), len(phio)))
        
        # # Storing the configuration corresponding to the minimum path length
        # min_dist = np.infty
        # thetai_min = 0
        # phii_min = 0
        # thetao_min = 0
        # phio_min = 0
        
        # # Computing path lengths for all the configurations and selecting the
        # # optimal configuration
        # for i in range(len(thetai)):
        #     for j in range(len(phii)):
        #         for k in range(len(thetao)):
        #             for l in range(len(phio)):
                        
        #                 # Obtaining the configuration of the Dubins vehicle
        #                 # at point of entry and exit from the cylinder
        #                 Tic, Toc, Pic, Poc, TicB, TocB, PicB, PocB, _ = \
        #                     configurations_discrete_angles(thetai[i], phii[j], thetao[k], phio[l], ini_config_right_sp,\
        #                                                    xaxis_rr, axis_rr, ht_cyl_rr, R)
                            
        #                 # Finding the minimum distance on the first sphere
        #                 # Position of configurations on the sphere wrt origin of frame placed
        #                 # at the center of the initial sphere is passed     
        #                 if np.isnan(sp_1_rr_path_lengths[i, j]): # Checking if path length was already computed
                            
        #                     filename_sp = "sp_1_rr_thetai_" + str(i) + "_phii_" + str(j) + ".html"
        #                     sp_1_rr_path_lengths[i, j], _, _, _, _, _, _ = \
        #                         CustomInputFromOtherModule(filename_sp, 'p', ini_config[0, :] - ini_config_right_sp,\
        #                                                    Pic - ini_config_right_sp, ini_config[1, :], Tic, r, R, 10)
                        
        #                 # Finding the minimum distance on the cylinder
        #                 # For this purpose, the configurations in the frame fixed
        #                 # at the origin of the cylinder is passed
        #                 if np.isnan(cyl_rr_path_lengths[j, k, l]): # Checking if path length was already computed
                        
        #                     filename_cyc = "cyc_rr_phii_" + str(j) + "_thetao_" + str(k) + "_phio_" + str(l) + ".html"
        #                     # cyl_rr_path_lengths[j, k, l], _, _ = generate_visualize_path_simple(PicB, TicB, R, PocB, TocB,\
        #                     #            ht_cyl_rr, 2, r, filename_cyc)
        #                     cyl_rr_path_lengths[j, k, l], _, _ = generate_visualize_path(PicB, TicB, R, PocB, TocB,\
        #                                ht_cyl_rr, 1, r, filename_cyc)
                        
        #                 # Finding the minimum distance on the last sphere
        #                 # Position of configurations on the sphere wrt origin of frame placed
        #                 # at the center of the initial sphere is passed
        #                 if np.isnan(sp_2_rr_path_lengths[k, l]): # Checking if path length was already computed
                            
        #                     filename_sp = "sp_2_rr_thetao_" + str(k) + "_phio_" + str(l) + ".html"
        #                     sp_2_rr_path_lengths[k, l], _, _, _, _, _, _ = \
        #                         CustomInputFromOtherModule(filename_sp, 'p', Poc - fin_config_right_sp,\
        #                                                    fin_config[0, :] - fin_config_right_sp, Toc, fin_config[1, :],\
        #                                                    r, R, 10)                       
        
        #                 path_lengths[i, j, k, l] = sp_1_rr_path_lengths[i, j] + \
        #                     cyl_rr_path_lengths[j, k, l] + sp_2_rr_path_lengths[k, l]
                            
        #                 # Checking if obtained solution is better
        #                 if path_lengths[i, j, k, l] < min_dist:
                            
        #                     min_dist = path_lengths[i, j, k, l]
        #                     thetai_min = thetai[i]
        #                     phii_min = phii[j]
        #                     thetao_min = thetao[k]
        #                     phio_min = phio[l]
        
        # # Obtaining the configuration of the Dubins vehicle
        # # at point of entry and exit from the cylinder corresponding to the
        # # best discretization
        # Tic, Toc, Pic, Poc, TicB, TocB, PicB, PocB, R_comp = \
        #     configurations_discrete_angles(thetai_min, phii_min, thetao_min, phio_min, ini_config_right_sp,\
        #                                    xaxis_rr, axis_rr, ht_cyl_rr, R)
        
        # # Obtaining the variables for plotting
        # # Obtaining variables for plotting on the first sphere
        # filename_sp = "sp_1_rr_optimal.html"
        # minlen_sp1, mintype_sp1, minFig_sp1, _, _, _, _ =\
        #     CustomInputFromOtherModule(filename_sp, 'p', ini_config[0, :] - ini_config_right_sp,\
        #                                Pic - ini_config_right_sp, ini_config[1, :], Tic, r, R, 50)
        # # Obtaining variables for plotting on the cylinder
        # filename_cyc = "cyc_rr_optimal.html"
        # # minlen_cyc, mintype_cyc, points_body_cyc = generate_visualize_path_simple(PicB, TicB, R, PocB, TocB,\
        # #                                                                  ht_cyl_rr, 2, r, filename_cyc)
        # minlen_cyc, mintype_cyc, points_body_cyc = generate_visualize_path(PicB, TicB, R, PocB, TocB,\
        #                                                                  ht_cyl_rr, 1, r, filename_cyc)
        # # Obtaining variables for plotting on the last sphere
        # filename_sp = "sp_2_rr_optimal.html"
        # minlen_sp2, mintype_sp2, minFig_sp2, _, _, _, _ =\
        #     CustomInputFromOtherModule(filename_sp, 'p', Poc - fin_config_right_sp,\
        #                                fin_config[0, :] - fin_config_right_sp, Toc, fin_config[1, :],\
        #                                r, R, 50)
                
        # # Finding the global points of the path on the cylinder using a coordinate
        # # transformation
        # points_global_cyc = np.empty(np.shape(points_body_cyc))
        # for i in range(np.shape(points_body_cyc)[0]): # iterating through all points which are present row-wise
        
        #     points_global_cyc[i, :] = np.matmul(R_comp, points_body_cyc[i, :]) + ini_config_right_sp
        
        # # Adding the plots for the optimal configuration
        # plot_figure.arrows_3D([Pic[0]], [Pic[1]], [Pic[2]], [Tic[0]], [Tic[1]], [Tic[2]],\
        #                       'brown', 'brwnyl', False)
        # plot_figure.arrows_3D([Poc[0]], [Poc[1]], [Poc[2]], [Toc[0]], [Toc[1]], [Toc[2]],\
        #                       'brown', 'brwnyl', False)        
        
        # # Updating the figure with path on the first sphere
        # PP_modified(plot_figure, 'Optimal path', [0], minFig_sp1[2], minFig_sp1[3], [0], ini_config_right_sp)        
        # # Plotting the path on the cylinder
        # plot_figure.scatter_3D(points_global_cyc[:, 0], points_global_cyc[:, 1],\
        #                        points_global_cyc[:, 2], 'blue', False)
        # # Updating the figure with path on the first sphere
        # PP_modified(plot_figure, 'Optimal path', [0], minFig_sp2[2], minFig_sp2[3], [0], fin_config_right_sp)
                        
        # # Writing the figure on the html file
        # plot_figure.writing_fig_to_html(filename, 'a')
        
        # # Writing onto the file
        # str_write = ["\n"]
        # str_write.append('Total length of the feasible path is ' + str(min_dist) + '.')
        # str_write.append('Angles corresponding to position of entry and exit from the cylinder are ' + str(thetai_min*180/math.pi)\
        #                  + ' and ' + str(thetao_min*180/math.pi) + ' degrees, respectively.')
        # str_write.append('Angles corresponding to headings at entry and exit from the cylinder are ' + str(phii_min*180/math.pi)\
        #                  + ' and ' + str(phio_min*180/math.pi) + ' degrees, respectively.')
        # str_write.append('Feasible path has a length of ' + str(minlen_sp1) + ' on the first sphere, ' +\
        #              str(minlen_cyc) + ' on the cylinder, and ' + str(minlen_sp2) + ' on the final sphere.')
        # str_write.append('Feasible path type on first sphere is ' + mintype_sp1 + '.')
        # str_write.append('Feasible path type on cylinder is ' + mintype_cyc.upper() + '.')
        # str_write.append('Feasible path type on the final sphere is ' + mintype_sp2 + '.')
        # with open(filename, 'a') as f:
        #     for i in range(len(str_write)):
                
        #         if(str_write[i] == "\n"):
                    
        #             f.write("<br />")
                
        #         else:
                    
        #             f.write(str_write[i] + "<br />")
        
    # return np.array([ini_config_left_sp, ini_config_right_sp, fin_config_left_sp, fin_config_right_sp]),\
    #     np.array([axis_ll, axis_lr, axis_rl, axis_rr]), np.array([ht_cyl_ll, ht_cyl_lr, ht_cyl_rl, ht_cyl_rr]),\
    #     sp_1_rr_path_lengths, cyl_rr_path_lengths, sp_2_rr_path_lengths

def configurations_discrete_angles(thetai, phii, thetao, phio, orig_cyl, xaxis_cyl, zaxis_cyl, ht_cyl, R):
    '''
    

    Parameters
    ----------
    thetai : TYPE
        DESCRIPTION.
    phii : TYPE
        DESCRIPTION.
    thetao : TYPE
        DESCRIPTION.
    phio : TYPE
        DESCRIPTION.
    orig_cyl : TYPE
        DESCRIPTION.
    xaxis_cyl : TYPE
        DESCRIPTION.
    zaxis_cyl : TYPE
        DESCRIPTION.
    ht_cyl : TYPE
        DESCRIPTION.
    R : TYPE
        DESCRIPTION.

    Returns
    -------
    Tic : TYPE
        DESCRIPTION.
    Toc : TYPE
        DESCRIPTION.
    Pic : TYPE
        DESCRIPTION.
    Poc : TYPE
        DESCRIPTION.
    TicB : TYPE
        DESCRIPTION.
    TocB : TYPE
        DESCRIPTION.
    PicB : TYPE
        DESCRIPTION.
    PocB : TYPE
        DESCRIPTION.
    Rcomposite : TYPE
        DESCRIPTION.

    '''
    
    # y-axis of the body frame
    yaxis_cyl = np.cross(zaxis_cyl, xaxis_cyl)
    
    # Initial and final tangent vectors in the body frame
    TicB = np.array([-math.sin(thetai)*math.cos(phii), math.cos(thetai)*math.cos(phii), math.sin(phii)])
    TocB = np.array([-math.sin(thetao)*math.cos(phio), math.cos(thetao)*math.cos(phio), math.sin(phio)])
    
    # Initial and final positions in the body frame
    PicB = np.array([R*math.cos(thetai), R*math.sin(thetai), 0])
    PocB = np.array([R*math.cos(thetao), R*math.sin(thetao), ht_cyl])
    
    # Initial and final tangent vectors in the global frame
    Rcomposite = np.array([[xaxis_cyl[0], yaxis_cyl[0], zaxis_cyl[0]],\
                           [xaxis_cyl[1], yaxis_cyl[1], zaxis_cyl[1]],\
                           [xaxis_cyl[2], yaxis_cyl[2], zaxis_cyl[2]]])
    Tic = np.matmul(Rcomposite, TicB)
    Toc = np.matmul(Rcomposite, TocB)
    
    # Initial and final positions in the global frame
    Pic = np.matmul(Rcomposite, PicB) + orig_cyl
    Poc = np.matmul(Rcomposite, PocB) + orig_cyl
    
    return Tic, Toc, Pic, Poc, TicB, TocB, PicB, PocB, Rcomposite
        
def generate_points_sphere(center, R):
    '''
    This function generates points on a sphere whose center is given by the variable
    "center" and with a radius of R.

    Parameters
    ----------
    center : Numpy 1x3 array
        Contains the coordinates corresponding to the center of the sphere.
    R : Scalar
        Contains the radius of the sphere.

    Returns
    -------
    x_grid : Numpy nd array
        Contains the x-coordinate of the points on the sphere.
    y_grid : Numpy nd array
        Contains the y-coordinate of the points on the sphere.
    z_grid : Numpy nd array
        Contains the z-coordinate of the points on the sphere.

    '''
    
    theta = np.linspace(0, 2*np.pi, 50)
    phi = np.linspace(-np.pi/2, np.pi/2, 50)
    
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Finding the coordinates of the points on the sphere in the global frame
    x_grid = center[0] + R*np.cos(theta_grid)*np.cos(phi_grid)
    y_grid = center[1] + R*np.sin(theta_grid)*np.cos(phi_grid)
    z_grid = center[2] + R*np.sin(phi_grid)
    
    return x_grid, y_grid, z_grid
    
def generate_points_cylinder(start_pt, axis, R, height_axis):
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

    Raises
    ------
    Exception
        Exception is raised when the random vector generated for x-axis of the body
        frame is along the axis of the cylinder. Though this event occuring has a very
        less probability, this is included just in case.

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
    
    # Defining the rotation matrix for transforming the cylinder
    # with the origin as the center and the z-axis as its axis to start_pt and with
    # axis variable as its axis
    # Declaring a random vector and applying Gram-Schmidt to obtain the direction
    # cosines of the x and y-axis
    temp = np.random.rand(3)
    # Checking that the generated random vector is not linear dependent with axis
    if np.linalg.norm(temp - np.dot(temp, axis)*axis) >= 0.01:
        
       x_dc =  (temp - np.dot(temp, axis)*axis)/np.linalg.norm(temp - np.dot(temp, axis)*axis)
   
    else:
       
       raise Exception('Regenerate the random vector')
   
    # Finding the direction cosines of the y-axis
    y_dc = np.cross(axis, x_dc)
    
    # Defining the rotation matrix
    rot_mat = np.array([[x_dc[0], y_dc[0], axis[0]],\
                  [x_dc[1], y_dc[1], axis[1]],\
                  [x_dc[2], y_dc[2], axis[2]]])
        
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