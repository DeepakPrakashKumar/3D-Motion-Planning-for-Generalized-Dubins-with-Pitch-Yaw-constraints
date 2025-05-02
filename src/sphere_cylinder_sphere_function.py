import numpy as np
import math
import copy
import os
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
from Path_generation_sphere import optimal_path_sphere, generate_points_sphere

# Returning to initial directory
os.chdir(cwd)

def Path_generation_sphere_cylinder_sphere(ini_config, fin_config, center_ini_sphere, center_fin_sphere,\
                                           r, Ryaw, Rpitch, axis_cylinder, ht_cylinder, disc_no_loc, disc_no_heading, plot_figure_configs,\
                                           visualization = 1, filename = "temp.html", type = 'inner', vis_int = 0):
    '''
    In this function, the paths connecting a given pair of spheres (inner or outer) with
    a cylindrical envelope is generated.

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
    r_min: Scalar
        Radius of the tight turn on sphere.
    Ryaw, Rpitch : Scalar
        Minimum turning radius for pitch and yaw motion.
    axis_cylinder : Array
        Axis of the cylinder connecting the considered pair of spheres.
    ht_cylinder : Scalar
        Length of the cylinder connecting the considered pair of spheres.
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
    type : String
        Type of sphere connections considered, i.e., inner-inner or outer-outer.
    vis_int : Scalar, optional
        Variable to decide whether to show the paths for intermediary calculations.
        
    Returns
    -------

    '''

    # Discretizing the initial and final angles.
    # NOTE: thetai and thetao are generated such that they are in the interval
    # [0, 2pi). Therefore, they cannot take the value of 2pi, as this will then
    # cause a redundancy.
    thetai = np.linspace(0, 2*math.pi, disc_no_loc, endpoint = False)
    thetao = np.linspace(0, 2*math.pi, disc_no_loc, endpoint = False)
    phii = np.linspace(0, math.pi, disc_no_heading)
    phio = np.linspace(0, math.pi, disc_no_heading)

    # We generate a random vector and orthonormalize it with respect to the axis
    # of the cylinder to obtain the x-axis for the cylinder.
    flag = 0; counter = 0
    tol = 10**(-2) # tolerance for the dot product
    # while flag == 0:

    #     # Generating a random vector
    #     temp = np.random.rand(3)

    #     # Orthonormalizing using Gram Schmidt
    #     if np.linalg.norm(-np.dot(temp, axis_cylinder)*axis_cylinder + temp) > tol:

    #         # In this case, we have obtained the desired x-axis
    #         x = (-np.dot(temp, axis_cylinder)*axis_cylinder + temp)\
    #             /np.linalg.norm(-np.dot(temp, axis_cylinder)*axis_cylinder + temp)
            
    #         flag = 1

    #     else:
            
    #         # We check if we have exceeded a threshold counter to ensure that we do not
    #         # go into an infinite loop
    #         if counter > 5:
    #             raise Exception('Going into an infinite loop for generating the random vector')
            
    #         # Incrementing the counter
    #         counter += 1

    # We consider the x, y, and z vectors and consider to orthonoramlize them
    vect_arr = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    while flag == 0:

        vector = vect_arr[counter]

        if np.linalg.norm(-np.dot(vector, axis_cylinder)*axis_cylinder + vector) > tol:

            # In this case, we have obtained the desired x-axis
            x = (-np.dot(vector, axis_cylinder)*axis_cylinder + vector)\
                /np.linalg.norm(-np.dot(vector, axis_cylinder)*axis_cylinder + vector)
            
            flag = 1

        else:

            if counter < 2:
                counter += 1
            else:
                raise Exception('Going into an infinite loop for generating the random vector')

    # For the surface normal, depending on whether an inner-inner or outer-outer connection is considered, the sign will differ
    if type == 'outer': sign = 1; lrsign = 0; R = Rpitch; r_cyc = Ryaw
    elif type == 'inner': sign = -1; lrsign = 0; R = Rpitch; r_cyc = Ryaw
    elif type == 'left': lrsign = 1; sign = 0; R = Ryaw; r_cyc = Rpitch
    else: lrsign = -1; sign = 0; R = Ryaw; r_cyc = Rpitch

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
    
        # Plotting the cylinders connecting the spheres
        x_cyl_conn_type, y_cyl_conn_type, z_cyl_conn_type =\
            generate_points_cylinder(center_ini_sphere, axis_cylinder, R, ht_cylinder, x)
        plot_figure.surface_3D(x_cyl_conn_type, y_cyl_conn_type, z_cyl_conn_type, 'lightgreen', 'Cylinder', 0.4)

        # Plotting the xaxis for the cylinder
        plot_figure.arrows_3D([center_ini_sphere[0]], [center_ini_sphere[1]], [center_ini_sphere[2]],\
                              [axis_cylinder[0]], [axis_cylinder[1]], [axis_cylinder[2]], 'grey', 'greys',\
                              False, 5, 5, 4, 'n')
            
        plot_figure.update_layout_3D('X (m)', 'Y (m)', 'Z (m)',\
                                    'Visualization of surfaces connecting initial and final configurations' +\
                                    ' via cylindrical envelope connecting ' + type + ' spheres')
        
        # Writing the figure on the html file
        plot_figure.writing_fig_to_html(filename, 'a')

    # Defining empty arrays to hold the path length for each discretization set
    sp_1_path_lengths = np.empty((len(thetai), len(phii)))
    sp_1_path_lengths[:] = np.NaN
    sp_2_path_lengths = np.empty((len(thetao), len(phio)))
    sp_2_path_lengths[:] = np.NaN
    cyl_path_lengths = np.empty((len(phii), len(thetao), len(phio))) # Note that for cylinder, though it depends on
    # four parameters, thetao and thetai both do not influence it; rather, the Delta theta influences the path length.
    cyl_path_lengths[:] = np.NaN

    # Final array to store path lengths for all configurations        
    path_lengths = np.empty((len(thetai), len(phii), len(thetao), len(phio)))
    
    # Storing the configuration corresponding to the minimum path length
    min_dist = np.infty
    thetai_min = 0
    phii_min = 0
    thetao_min = 0
    phio_min = 0

    # We now run through each discretization and generate paths on the spheres and cylinder
    # Computing path lengths for all the configurations and selecting the
    # optimal configuration
    for i in range(len(thetai)):
        for j in range(len(phii)):
            for k in range(len(thetao)):
                for l in range(len(phio)):

                    # Obtaining the configuration of the Dubins vehicle
                    # at point of entry and exit from the cylinder
                    Tic, Toc, Pic, Poc, TicB, TocB, PicB, PocB, _ = \
                        configurations_discrete_angles(thetai[i], phii[j], thetao[k], phio[l],\
                                                       center_ini_sphere, x, axis_cylinder, ht_cylinder, R)
                    
                    # Computing the path length for the two spheres
                    if np.isnan(sp_1_path_lengths[i, j]): # Checking if path length was already computed
                        
                        # We need to pass the initial and final configuration such that they are with
                        # respect to the center of the sphere
                        # Constructing the initial configuration for the sphere
                        ini_config_sphere = config_sphere(ini_config[0, :], center_ini_sphere, ini_config[1, :])

                        # Constructing the intermediary final configuration
                        # Now, the final configuration for the first sphere is constructed
                        fin_config_sphere = config_sphere(Pic, center_ini_sphere, Tic)

                        # Obtaining the optimal path on the initial sphere. The plot is generated only if visualization = 1. The
                        # same is the case for path on the cylindrical envelope and the final sphere.
                        filename_sp = "sp_1_thetai_" + str(i) + "_phii_" + str(j) + ".html"

                        sp_1_path_lengths[i, j] =\
                            optimal_path_sphere(ini_config_sphere, fin_config_sphere, r, R, vis_int, path_config = vis_int, filename = filename_sp)[1]
                        
                    if np.isnan(sp_2_path_lengths[k, l]): # Checking if path length was already computed
                        
                        # We need to pass the initial and final configuration such that they are with
                        # respect to the center of the sphere
                        # Constructing the initial configuration for the sphere
                        ini_config_sphere = config_sphere(Poc, center_fin_sphere, Toc)
                        
                        # Now, the final configuration for the second sphere is constructed
                        fin_config_sphere = config_sphere(fin_config[0, :], center_fin_sphere, fin_config[1, :])

                        # Obtaining the optimal path on the initial sphere
                        filename_sp = "sp_2_thetao_" + str(k) + "_phio_" + str(k) + ".html"

                        sp_2_path_lengths[k, l] =\
                            optimal_path_sphere(ini_config_sphere, fin_config_sphere, r, R, vis_int, path_config = vis_int, filename = filename_sp)[1]
                        
                    # Now, we construct the path on the cylinder
                    k_mod = np.mod(k - i, len(thetao))
                    if np.isnan(cyl_path_lengths[j, k_mod, l]): # Checking if path length was already computed
                        
                        filename_cyc = "cyc_phii_" + str(j) + "_thetao_" + str(k_mod) + "_phio_" + str(l) + ".html"

                        cyl_path_lengths[j, k_mod, l] = generate_visualize_path(PicB, TicB, R, PocB, TocB,\
                                    ht_cylinder, vis_int, r_cyc, path_config = 0, filename = filename_cyc)[0]
                        
                    # cost = generate_visualize_path(PicB, TicB, R, PocB, TocB,\
                    #                 ht_cylinder, vis_int, r_cyc, filename_cyc)[0]
                    # print('i = ', i, 'j = ', j, 'k = ', k, 'l = ', l, cost)
                    # print('Cost considered is ', cyl_path_lengths[j, k_mod, l])
                    # if abs(cost - cyl_path_lengths[j, k_mod, l]) > 1e-6:
                    #     raise Exception('Exceeded tolerance')

                    path_lengths[i, j, k, l] = sp_1_path_lengths[i, j] + cyl_path_lengths[j, k_mod, l] + sp_2_path_lengths[k, l]
                            
                    # Checking if obtained solution is better
                    if path_lengths[i, j, k, l] < min_dist:
                        
                        min_dist = path_lengths[i, j, k, l]; thetai_min = thetai[i]; phii_min = phii[j]; thetao_min = thetao[k]
                        phio_min = phio[l]

    # print(cyl_path_lengths)
    # raise Exception('Exit')

    # We now plot the optimal path and/or obtain the configurations along the path
    # Obtaining the configuration of the Dubins vehicle
    # at point of entry and exit from the cylinder corresponding to the
    # best discretization
    Tic, Toc, Pic, Poc, TicB, TocB, PicB, PocB, R_comp = \
        configurations_discrete_angles(thetai_min, phii_min, thetao_min, phio_min,\
                                       center_ini_sphere, x, axis_cylinder, ht_cylinder, R)
    
    # Obtaining the variables for plotting
    # Obtaining variables for plotting on the first sphere
    filename_sp = "sp_1_optimal.html"    
    # Obtaining the configuration on the first sphere
    ini_config_sphere = config_sphere(ini_config[0, :], center_ini_sphere, ini_config[1, :])
    fin_config_sphere = config_sphere(Pic, center_ini_sphere, Tic)
    # Obtaining the best feasible path's portion on the first sphere
    _, _, _, minlen_sp1_path_points_x, minlen_sp1_path_points_y, minlen_sp1_path_points_z, minlen_sp1_Tx, minlen_sp1_Ty, minlen_sp1_Tz =\
        optimal_path_sphere(ini_config_sphere, fin_config_sphere, r, R, vis_int, path_config = 1, filename = filename_sp)[0:9]
    
    # Obtaining variables for plotting on the cylinder
    filename_cyc = "cyc_optimal.html"    
    _, _, points_body_cyc, tang_body_cyc, norm_vect_body_cyc = generate_visualize_path(PicB, TicB, R, PocB, TocB,\
                                                                        ht_cylinder, vis_int, r_cyc, path_config = 1, filename = filename_cyc)
    
    # Obtaining variables for plotting on the last sphere
    filename_sp = "sp_2_optimal.html"
    # Obtaining the configuration on the last sphere
    ini_config_sphere = config_sphere(Poc, center_fin_sphere, Toc)
    fin_config_sphere = config_sphere(fin_config[0, :], center_fin_sphere, fin_config[1, :])
    # Obtaining the best feasible path's portion on the second sphere
    _, _, _, minlen_sp2_path_points_x, minlen_sp2_path_points_y, minlen_sp2_path_points_z, minlen_sp2_Tx, minlen_sp2_Ty, minlen_sp2_Tz =\
        optimal_path_sphere(ini_config_sphere, fin_config_sphere, r, R, vis_int, path_config = 1, filename = filename_sp)[0:9]
            
    # Finding the global points of the path on the first sphere using a coordinate transformation
    points_global = np.empty((len(minlen_sp1_path_points_x) + len(points_body_cyc) + len(minlen_sp2_path_points_x), 3))
    tang_global = np.empty((len(minlen_sp1_path_points_x) + len(points_body_cyc) + len(minlen_sp2_path_points_x), 3))
    tang_normal_global = np.empty((len(minlen_sp1_path_points_x) + len(points_body_cyc) + len(minlen_sp2_path_points_x), 3))
    surf_normal_global = np.empty((len(minlen_sp1_path_points_x) + len(points_body_cyc) + len(minlen_sp2_path_points_x), 3))

    for i in range(len(minlen_sp1_path_points_x)):

        points_global[i, 0] = minlen_sp1_path_points_x[i] + center_ini_sphere[0]
        points_global[i, 1] = minlen_sp1_path_points_y[i] + center_ini_sphere[1]
        points_global[i, 2] = minlen_sp1_path_points_z[i] + center_ini_sphere[2]
        tang_global[i, 0] = minlen_sp1_Tx[i]; tang_global[i, 1] = minlen_sp1_Ty[i]; tang_global[i, 2] = minlen_sp1_Tz[i]
        
        if sign != 0: # In this case, inner and outer spheres have been considered

            surf_normal_global[i, 0] = sign*minlen_sp1_path_points_x[i]/R;
            surf_normal_global[i, 1] = sign*minlen_sp1_path_points_y[i]/R;
            surf_normal_global[i, 2] = sign*minlen_sp1_path_points_z[i]/R;
            tang_normal_global[i, :] = np.cross(surf_normal_global[i], tang_global[i])

        else:

            tang_normal_global[i, 0] = -lrsign*minlen_sp1_path_points_x[i]/R;
            tang_normal_global[i, 1] = -lrsign*minlen_sp1_path_points_y[i]/R;
            tang_normal_global[i, 2] = -lrsign*minlen_sp1_path_points_z[i]/R;
            surf_normal_global[i, :] = np.cross(tang_global[i], tang_normal_global[i])

    # Finding the global points of the path on the cylinder using a coordinate
    # transformation
    sp1_pts_length = len(minlen_sp1_path_points_x)
    for i in range(np.shape(points_body_cyc)[0]): # iterating through all points which are present row-wise
    
        ind = i + sp1_pts_length
        points_global[ind, :] = np.matmul(R_comp, points_body_cyc[i, :]) + center_ini_sphere
        # Determining the expressions for the tangent vector, tangent normal, and surface normal vectors on the cylinder
        tang_global[ind, :] = np.matmul(R_comp, tang_body_cyc[i, :])
        # surf_normal_global[ind, :] = np.matmul(R_comp, sign*norm_vect_body_cyc[i, :])
        # tang_normal_global[ind, :] = np.cross(surf_normal_global[ind], tang_global[ind])

        if sign != 0:

            surf_normal_global[ind, :] = np.matmul(R_comp, sign*norm_vect_body_cyc[i, :])
            tang_normal_global[ind, :] = np.cross(surf_normal_global[ind], tang_global[ind])

        else:

            tang_normal_global[ind, :] = np.matmul(R_comp, -lrsign*norm_vect_body_cyc[i, :])
            surf_normal_global[ind, :] = np.cross(tang_global[ind], tang_normal_global[ind])

    # Finding the global points of the path on the last sphere using a coordinate transformation
    sp1_pls_cyl_points_length = len(points_body_cyc) + sp1_pts_length
    for i in range(len(minlen_sp2_path_points_x)):

        ind = i + sp1_pls_cyl_points_length
        points_global[ind, 0] = minlen_sp2_path_points_x[i] + center_fin_sphere[0]
        points_global[ind, 1] = minlen_sp2_path_points_y[i] + center_fin_sphere[1]
        points_global[ind, 2] = minlen_sp2_path_points_z[i] + center_fin_sphere[2]
        tang_global[ind, 0] = minlen_sp2_Tx[i]; tang_global[ind, 1] = minlen_sp2_Ty[i]; tang_global[ind, 2] = minlen_sp2_Tz[i]
        
        # surf_normal_global[ind, 0] = sign*minlen_sp2_path_points_x[i]/R;
        # surf_normal_global[ind, 1] = sign*minlen_sp2_path_points_y[i]/R;
        # surf_normal_global[ind, 2] = sign*minlen_sp2_path_points_z[i]/R;
        # tang_normal_global[ind, :] = np.cross(surf_normal_global[ind], tang_global[ind])
        if sign != 0: # In this case, inner and outer spheres have been considered

            surf_normal_global[ind, 0] = sign*minlen_sp2_path_points_x[i]/R;
            surf_normal_global[ind, 1] = sign*minlen_sp2_path_points_y[i]/R;
            surf_normal_global[ind, 2] = sign*minlen_sp2_path_points_z[i]/R;
            tang_normal_global[ind, :] = np.cross(surf_normal_global[ind], tang_global[ind])

        else:

            tang_normal_global[ind, 0] = -lrsign*minlen_sp2_path_points_x[i]/R;
            tang_normal_global[ind, 1] = -lrsign*minlen_sp2_path_points_y[i]/R;
            tang_normal_global[ind, 2] = -lrsign*minlen_sp2_path_points_z[i]/R;
            surf_normal_global[ind, :] = np.cross(tang_global[ind], tang_normal_global[ind])
    
    if visualization == 1:
    
        # Adding the plots for the optimal configuration
        plot_figure.arrows_3D([Pic[0]], [Pic[1]], [Pic[2]], [Tic[0]], [Tic[1]], [Tic[2]],\
                                'brown', 'brwnyl', False)
        plot_figure.arrows_3D([Poc[0]], [Poc[1]], [Poc[2]], [Toc[0]], [Toc[1]], [Toc[2]],\
                                'brown', 'brwnyl', False)
        
        # Plotting the path
        plot_figure.scatter_3D(points_global[:, 0], points_global[:, 1], points_global[:, 2], 'blue', 'Optimal path')

        # Plotting the configuration along the path
        # print('Plotting the configuration along the path')
        for i in range(len(points_global)):

            if np.mod(i, 40) == 39:

                plot_figure.arrows_3D([points_global[i, 0]], [points_global[i, 1]], [points_global[i, 2]],\
                                        [tang_global[i, 0]], [tang_global[i, 1]], [tang_global[i, 2]],\
                                        'orange', 'oranges', False, 5, 5, 4, 'n')
                plot_figure.arrows_3D([points_global[i, 0]], [points_global[i, 1]], [points_global[i, 2]],\
                                        [tang_normal_global[i, 0]], [tang_normal_global[i, 1]], [tang_normal_global[i, 2]],\
                                        'purple', 'purp', False, 5, 5, 4, 'n')
                plot_figure.arrows_3D([points_global[i, 0]], [points_global[i, 1]], [points_global[i, 2]],\
                                        [surf_normal_global[i, 0]], [surf_normal_global[i, 1]], [surf_normal_global[i, 2]],\
                                        'green', 'greens', False, 5, 5, 4, 'n')
                        
        plot_figure.update_layout_3D('X (m)', 'Y (m)', 'Z (m)',\
                                    'Best feasible path connecting ' + type + ' spheres using a cylindrical envelope')

        # Writing the figure on the html file
        plot_figure.writing_fig_to_html(filename, 'a')

    return min_dist, points_global, tang_global, tang_normal_global, surf_normal_global

def configurations_discrete_angles(thetai, phii, thetao, phio, orig_cyl, xaxis_cyl, zaxis_cyl, ht_cyl, R):
    '''
    In this function, the configurations as a function of the parameters selected is computed for the inner-inner
    and outer-outer sphere combinations connected through a cylindrical envelope.

    Parameters
    ----------
    thetai, phii, thetao, phio : Scalars
        Angles depicting the entry location at the base of the cylider (thetai) with heading angle phii,
        and exit location at the base of the cylinder (thetao) with heading angle phio.
    orig_cyl : Array
        Contains the coordinates of the origin of the cylinder.
    xaxis_cyl : Array
        Contains the direction cosines of the x-axis of the cylinder.
    zaxis_cyl : Array
        Contains the direction cosines for the axis of the cylinder.
    ht_cyl : Scalar
        Height of the cylinder.
    R : Scalar
        Radius of the spheres and cylinder.

    Returns
    -------
    Tic, Toc : Arrays
        Direction cosines of the tangent vector at the entry and exit of the cylinder in the global frame.
    Pic, Poc : Arrays
        Arrays containing the global position for entry and exit at the cylinder.
    TicB, TocB : Arrays
        Direction cosines of the tangent vector at the entry and exit of the cylinder in the body frame.
    PicB, PocB : Arrays
        Position for the entry and exit at the cylinder in the body frame.
    Rcomposite : Array
        Contains the rotation matrix between the global frame and body frame.

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