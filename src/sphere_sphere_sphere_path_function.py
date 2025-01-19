import numpy as np
import math
import copy
import os
import sys
from math import cos as cos
from math import sin as sin
from math import sqrt as sqrt

# Including the following command to ensure that python is able to find the relevant files afer changing directory
sys.path.insert(0, '')
# Obtaining the current directory
cwd = os.getcwd()

from sphere_cylinder_sphere_function import config_sphere

# Importing code for the sphere
rel_path = '\Sphere code'
os.chdir(cwd + rel_path)
from Path_generation_sphere import optimal_path_sphere_three_seg, generate_points_sphere

# Returning to initial directory
os.chdir(cwd)

def Path_generation_sphere_sphere_sphere(ini_config, fin_config, center_ini_sphere, center_fin_sphere,\
                                           r, R, axis_plane, dist_center_spheres, disc_no, plot_figure_configs,\
                                           visualization = 1, filename = "temp.html", vis_int = 0):
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
    dist_center_spheres : Scalar
        Distance between the centers of the initial and final spheres.
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

    # We first check for existence of such a connection depending on the distance between
    # the centers of the spheres
    if dist_center_spheres > 4*R:
        return np.NaN, []

    # Discretizing the angle for parameterizing the intermediary sphere and the
    # angles for the tangent vector for exit from initial sphere and entry into final
    # sphere
    theta = np.linspace(0, 2*math.pi, disc_no, endpoint = False)
    thetaic = np.linspace(0, 2*math.pi, disc_no, endpoint = False)
    thetafc = np.linspace(0, 2*math.pi, disc_no, endpoint = False)

    # Obtaining the angle to describe the locus of the intermediary sphere
    phi = math.acos(dist_center_spheres/(4*R))

    # Generating a random vector x perpendicular to the vector connecting the centers of the
    # initial and final spheres.
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

    # Plotting the configurations and spheres if visualization is 1.
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
                                    ' via sphere')
        
        # Writing the figure on the html file
        plot_figure.writing_fig_to_html(filename, 'a')

    sp1_path_lengths = np.empty((len(theta), len(thetaic)))
    sp1_path_lengths[:] = np.nan
    spint_path_lengths = np.empty((len(theta), len(thetaic), len(thetafc)))
    spint_path_lengths[:] = np.nan
    sp2_path_lengths = np.empty((len(theta), len(thetafc)))
    sp2_path_lengths[:] = np.nan
    path_lengths = np.empty((len(theta), len(thetaic), len(thetafc)))
    path_lengths[:] = np.nan

    # We run through each discretization
    y = np.cross(axis_plane, x)
    for (i, theta_val) in enumerate(theta):

        # We now obtain the expression for the center of the intermediary sphere
        Xcthetac = center_ini_sphere + 0.5*(center_fin_sphere - center_ini_sphere)\
            + 2*R*sin(phi)*(cos(theta_val)*x + sin(theta_val)*y)
        
        # print('Distance of center of sphere from initial sphere is ', np.linalg.norm(Xcthetac - center_ini_sphere))
        # print('Distance of center of sphere from final sphere is ', np.linalg.norm(Xcthetac - center_fin_sphere))
        
        # Now, we obtain the location corresponding to exit from the initial sphere
        # and entry onto the final sphere
        Xic = 0.5*(center_ini_sphere + Xcthetac)
        Xfc = 0.5*(center_fin_sphere + Xcthetac)

        # We now obtain the vectors for parameterizing the tangent vector
        xic = axis_plane/sin(phi) - (Xic - center_ini_sphere)/(R*math.tan(phi))
        xfc = -axis_plane/sin(phi) - (Xfc - center_fin_sphere)/(R*math.tan(phi))

        # We run through for loops to compute the paths for the spheres
        for (j, thetaic_val) in enumerate(thetaic):

            # We obtain the tangent vector at the initial sphere
            Tic = cos(thetaic_val)*xic + sin(thetaic_val)*(1/R)*np.cross(Xic - center_ini_sphere, xic)

            # We compute the path on the initial sphere
            ini_config_sp1 = config_sphere(ini_config[0, :], center_ini_sphere, ini_config[1, :])
            fin_config_sp1 = config_sphere(Xic, center_ini_sphere, Tic)
            filename_sp = "sp_1_thetai_" + str(i) + "_thetaic_" + str(j) + ".html"
            # print('Location on sphere is ', ini_config[0, :], ' and center is ', center_ini_sphere, '. Norm is ', np.linalg.norm(ini_config[0, :] - center_ini_sphere))
            # print('Location on sphere is ', fin_config[0, :], ' and center is ', center_ini_sphere, '. Norm is ', np.linalg.norm(fin_config[0, :] - center_ini_sphere))
            
            sp1_path_lengths[i, j] =\
                  optimal_path_sphere_three_seg(ini_config_sp1, fin_config_sp1, r, R, vis_int, filename_sp)[1]

            for (k, thetafc_val) in enumerate(thetafc):

                # We obtain the tangent vector at the final sphere
                Tfc = cos(thetafc_val)*xfc + sin(thetafc_val)*(1/R)*np.cross(Xfc - center_fin_sphere, xfc)

                # We compute the path on the final sphere if not already computed
                ini_config_sp2 = config_sphere(Xfc, center_fin_sphere, Tfc)
                fin_config_sp2 = config_sphere(fin_config[0, :], center_fin_sphere, fin_config[1, :])
                filename_sp = "sp_2_thetai_" + str(i) + "_thetafc_" + str(k) + ".html"

                if np.isnan(sp2_path_lengths[i, k]):

                    sp2_path_lengths[i, k] =\
                          optimal_path_sphere_three_seg(ini_config_sp2, fin_config_sp2, r, R, vis_int, filename_sp)[1]
                    
                # We compute the path length on the intermediary sphere as well
                ini_config_spint = config_sphere(Xic, Xcthetac, Tic)
                fin_config_spint = config_sphere(Xfc, Xcthetac, Tfc)
                filename_sp = "sp_int_thetai_" + str(i) + "_thetaic_" + str(j) + "_thetafc_" + str(k) + ".html"

                spint_path_lengths[i, j, k] =\
                      optimal_path_sphere_three_seg(ini_config_spint, fin_config_spint, r, R, vis_int, filename_sp)[1]
                
    # We now pick the best path
    min_dist = np.inf
    theta_min = np.nan; thetaic_min = np.nan; thetafc_min = np.nan
    for i in range(len(theta)):
        for j in range(len(thetaic)):
            for k in range(len(thetafc)):

                path_lengths[i, j, k] = sp1_path_lengths[i, j] + sp2_path_lengths[i, k] + spint_path_lengths[i, j, k]
                if path_lengths[i, j, k] < min_dist:

                    min_dist = path_lengths[i, j, k]
                    theta_min = theta[i]; thetaic_min = thetaic[j]; thetafc_min = thetafc[k]

    # We obtain the points along the optimal path and plot it if needed
    # Obtaining the configuration for the spheres for exit from the first sphere and entry at final sphere
    Xcthetac = center_ini_sphere + 0.5*(center_fin_sphere - center_ini_sphere)\
            + 2*R*sin(phi)*(cos(theta_min)*x + sin(theta_min)*y)
    # Now, we obtain the location corresponding to exit from the initial sphere
    # and entry onto the final sphere
    Xic = 0.5*(center_ini_sphere + Xcthetac)
    Xfc = 0.5*(center_fin_sphere + Xcthetac)

    # We now obtain the vectors for parameterizing the tangent vector
    xic = axis_plane/sin(phi) - (Xic - center_ini_sphere)/(R*math.tan(phi))
    xfc = -axis_plane/sin(phi) - (Xfc - center_fin_sphere)/(R*math.tan(phi))

    # We obtain the tangent vector at the initial sphere
    Tic = cos(thetaic_min)*xic + sin(thetaic_min)*(1/R)*np.cross(Xic - center_ini_sphere, xic)
    # We obtain the tangent vector at the final sphere
    Tfc = cos(thetafc_min)*xfc + sin(thetafc_min)*(1/R)*np.cross(Xfc - center_fin_sphere, xfc)

    # Obtaining the optimal path on all three spheres
    # Initial sphere
    ini_config_sp1 = config_sphere(ini_config[0, :], center_ini_sphere, ini_config[1, :])
    fin_config_sp1 = config_sphere(Xic, center_ini_sphere, Tic)
    filename_sp = "sp_1_optimal" + ".html"
    _, _, _, minlen_sp1_path_points_x, minlen_sp1_path_points_y, minlen_sp1_path_points_z, minlen_sp1_Tx, minlen_sp1_Ty, minlen_sp1_Tz =\
            optimal_path_sphere_three_seg(ini_config_sp1, fin_config_sp1, r, R, vis_int, filename_sp)
    
    # Intermediary sphere
    ini_config_spint = config_sphere(Xic, Xcthetac, Tic)
    fin_config_spint = config_sphere(Xfc, Xcthetac, Tfc)
    filename_sp = "sp_int_optimal" + ".html"

    _, _, _, minlen_spint_path_points_x, minlen_spint_path_points_y, minlen_spint_path_points_z, minlen_spint_Tx, minlen_spint_Ty, minlen_spint_Tz =\
            optimal_path_sphere_three_seg(ini_config_spint, fin_config_spint, r, R, vis_int, filename_sp)
    
    # Final sphere
    ini_config_sp2 = config_sphere(Xfc, center_fin_sphere, Tfc)
    fin_config_sp2 = config_sphere(fin_config[0, :], center_fin_sphere, fin_config[1, :])
    filename_sp = "sp_2_optimal" + ".html"

    _, _, _, minlen_sp2_path_points_x, minlen_sp2_path_points_y, minlen_sp2_path_points_z, minlen_sp2_Tx, minlen_sp2_Ty, minlen_sp2_Tz =\
            optimal_path_sphere_three_seg(ini_config_sp2, fin_config_sp2, r, R, vis_int, filename_sp)
    
    # Finding the global points of the path on the first sphere using a coordinate transformation
    points_global = np.empty((len(minlen_sp1_path_points_x) + len(minlen_spint_path_points_x) + len(minlen_sp2_path_points_x), 3))

    for i in range(len(minlen_sp1_path_points_x)):

        points_global[i, 0] = minlen_sp1_path_points_x[i] + center_ini_sphere[0]
        points_global[i, 1] = minlen_sp1_path_points_y[i] + center_ini_sphere[1]
        points_global[i, 2] = minlen_sp1_path_points_z[i] + center_ini_sphere[2]

    # Finding the global points of the path on the intermediary sphere
    sp1_pts_length = len(minlen_sp1_path_points_x)

    for i in range(len(minlen_spint_path_points_x)):

        ind = sp1_pts_length + i
        points_global[ind, 0] = minlen_spint_path_points_x[i] + Xcthetac[0]
        points_global[ind, 1] = minlen_spint_path_points_y[i] + Xcthetac[1]
        points_global[ind, 2] = minlen_spint_path_points_z[i] + Xcthetac[2]

    # Finding global points of the path on the final sphere
    sp1_plus_int_pts_length = sp1_pts_length + len(minlen_spint_path_points_x)

    for i in range(len(minlen_sp2_path_points_x)):

        ind = sp1_plus_int_pts_length + i
        points_global[ind, 0] = minlen_sp2_path_points_x[i] + center_fin_sphere[0]
        points_global[ind, 1] = minlen_sp2_path_points_y[i] + center_fin_sphere[1]
        points_global[ind, 2] = minlen_sp2_path_points_z[i] + center_fin_sphere[2]

    if visualization == 1:

        # Plotting the path on the first sphere
        plot_figure.scatter_3D(points_global[:, 0], points_global[:, 1],\
                                points_global[:, 2], 'blue', 'Optimal path')

        # Plotting the intermediary sphere as well
        plot_figure.surface_3D(generate_points_sphere(Xcthetac, R)[0],\
                               generate_points_sphere(Xcthetac, R)[1],\
                               generate_points_sphere(Xcthetac, R)[2], 'grey',\
                               'Final sphere', 0.7)
                        
        # Writing the figure on the html file
        plot_figure.writing_fig_to_html(filename, 'a')

    return min_dist, points_global