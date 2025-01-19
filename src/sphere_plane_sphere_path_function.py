import numpy as np
import math
import copy
import os
import sys

# Including the following command to ensure that python is able to find the relevant files afer changing directory
sys.path.insert(0, '')
# Obtaining the current directory
cwd = os.getcwd()

from sphere_cylinder_sphere_function import config_sphere

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

def Path_generation_sphere_plane_sphere(ini_config, fin_config, center_ini_sphere, center_fin_sphere,\
                                           r, R, axis_plane, ht_plane, disc_no, plot_figure_configs,\
                                           visualization = 1, filename = "temp.html", vis_int = 0):
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
    vis_int : Scalar, optional
        Variable to decide whether to show the paths for intermediary calculations.

    Returns
    -------

    '''

    # We first check if the considered connections exist
    if 2*R > ht_plane: # In this case, the path does not exist
        return np.NaN, []

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

    # Plotting the configurations, spheres, and planes if visualization is 1.
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

            if vis_int == 0: filename_plane = False
            else: filename_plane = "plane_phii_" + str(i) + "_phio_" + str(j) + ".html"

            # We obtain the initial and final configuration on the plane
            ini_config_plane = np.array([0, 0, phii[i]])
            fin_config_plane = np.array([math.sqrt(ht_plane**2 - 4*R**2), 0, phio[j]])

            plane_path_lengths[i, j] = optimal_dubins_path(ini_config_plane, fin_config_plane, r, filename_plane)[0]

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

            # print('Location on sphere is ', loc_ini, ' and center is ', center_ini_sphere, '. Norm is ', np.linalg.norm(loc_ini - center_ini_sphere))
            ini_sphere_fin_config = config_sphere(loc_ini, center_ini_sphere, T_ini)
            
            filename_sp = "sp_1_thetai_" + str(i) + "_phii_" + str(j) + ".html"
            sp_1_path_lengths[i, j] =\
                  optimal_path_sphere_three_seg(ini_sphere_ini_config, ini_sphere_fin_config, r, R, vis_int, filename_sp)[1]

        for j in range(len(phio)):
            
            T_ini = math.cos(phio[j])*t + math.sin(phio[j])*(np.cross((loc_ini - center_ini_sphere)/R, t))

            # Now, we construct the configuration for planning on the final sphere
            fin_sphere_ini_config = config_sphere(loc_fin, center_fin_sphere, T_ini)

            # Now, we construct the configuration for exit on the final sphere
            fin_sphere_fin_config = config_sphere(fin_config[0, :], center_fin_sphere, fin_config[1, :])

            filename_sp = "sp_2_thetai_" + str(i) + "_phio_" + str(j) + ".html"
            sp_2_path_lengths[i, j] =\
                  optimal_path_sphere_three_seg(fin_sphere_ini_config, fin_sphere_fin_config, r, R, vis_int, filename_sp)[1]

    min_dist = np.inf
    thetai_min = np.nan; phii_min = np.nan; phio_min = np.nan
    for i in range(len(thetai)):
        for j in range(len(phii)):
            for k in range(len(phio)):

                path_lengths[i, j, k] = sp_1_path_lengths[i, j] + plane_path_lengths[j, k] + sp_2_path_lengths[i, k]
                if path_lengths[i, j, k] < min_dist:
                    
                    min_dist = path_lengths[i, j, k]
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
    _, _, _, minlen_sp1_path_points_x, minlen_sp1_path_points_y, minlen_sp1_path_points_z, minlen_sp1_Tx, minlen_sp1_Ty, minlen_sp1_Tz =\
        optimal_path_sphere_three_seg(ini_config_sphere, fin_config_sphere, r, R, vis_int, "sp1_optimal_cross_tangent.html")
    
    # Obtaining the optimal path on the final sphere
    ini_config_sphere = config_sphere(loc_fin, center_fin_sphere, T_fin)
    fin_config_sphere = config_sphere(fin_config[0, :], center_fin_sphere, fin_config[1, :])
    # Obtaining the best feasible path's portion on the second sphere
    _, _, _, minlen_sp2_path_points_x, minlen_sp2_path_points_y, minlen_sp2_path_points_z, minlen_sp2_Tx, minlen_sp2_Ty, minlen_sp2_Tz =\
        optimal_path_sphere_three_seg(ini_config_sphere, fin_config_sphere, r, R, vis_int, "sp2_optimal_cross_tangent.html")

    # Obtaining the optimal path on the plane
    ini_config_plane = np.array([0, 0, phii_min])
    fin_config_plane = np.array([math.sqrt(ht_plane**2 - 4*R**2), 0, phio_min])
    if vis_int == 1: filename_plane = 'optimal_path_cross_tangent_plane.html'
    else: filename_plane = False
    _, _, _, pts_x, pts_y = optimal_dubins_path(ini_config_plane, fin_config_plane, r, filename_plane)

    # Finding the global points of the path on the first sphere using a coordinate transformation
    # points_global_sp1 = np.empty((len(minlen_sp1_path_points_x), 3))
    points_global = np.empty((len(minlen_sp1_path_points_x) + len(pts_x) + len(minlen_sp2_path_points_x), 3))
    for i in range(len(minlen_sp1_path_points_x)):

        points_global[i, 0] = minlen_sp1_path_points_x[i] + center_ini_sphere[0]
        points_global[i, 1] = minlen_sp1_path_points_y[i] + center_ini_sphere[1]
        points_global[i, 2] = minlen_sp1_path_points_z[i] + center_ini_sphere[2]

    # Finding the global points of the path on the cross-tangent plane
    # points_global_plane = np.empty((len(pts_x), 3))
    sp1_pts_length = len(minlen_sp1_path_points_x)
    # We construct the rotation matrix relating the frame corresponding to the plane.
    rot_mat = np.array([[t[0], np.cross((loc_ini - center_ini_sphere)/R, t)[0], ((loc_ini - center_ini_sphere)/R)[0]],\
                        [t[1], np.cross((loc_ini - center_ini_sphere)/R, t)[1], ((loc_ini - center_ini_sphere)/R)[1]],\
                        [t[2], np.cross((loc_ini - center_ini_sphere)/R, t)[2], ((loc_ini - center_ini_sphere)/R)[2]]])
    
    for i in range(len(pts_x)):

        ind = i + sp1_pts_length
        pos = loc_ini + np.matmul(rot_mat, np.array([pts_x[i], pts_y[i], 0]))
        points_global[ind, 0] = pos[0]
        points_global[ind, 1] = pos[1]
        points_global[ind, 2] = pos[2]

    # Finding the global points of the path on the last sphere using a coordinate transformation
    # points_global_sp2 = np.empty((len(minlen_sp2_path_points_x), 3))
    sp1_pls_plane_points_length = len(pts_x) + sp1_pts_length
    for i in range(len(minlen_sp2_path_points_x)):

        ind = i + sp1_pls_plane_points_length
        points_global[ind, 0] = minlen_sp2_path_points_x[i] + center_fin_sphere[0]
        points_global[ind, 1] = minlen_sp2_path_points_y[i] + center_fin_sphere[1]
        points_global[ind, 2] = minlen_sp2_path_points_z[i] + center_fin_sphere[2]

    if visualization == 1:

        # Plotting the path on the first sphere
        plot_figure.scatter_3D(points_global[:, 0], points_global[:, 1],\
                                points_global[:, 2], 'blue', 'Optimal path')  
        # # Plotting the path on the cylinder
        # plot_figure.scatter_3D(points_global_plane[:, 0], points_global_plane[:, 1],\
        #                         points_global_plane[:, 2], 'blue', False)
        # # Updating the figure with path on the last sphere
        # plot_figure.scatter_3D(points_global_sp2[:, 0], points_global_sp2[:, 1],\
        #                         points_global_sp2[:, 2], 'blue', False)
                        
        # Writing the figure on the html file
        plot_figure.writing_fig_to_html(filename, 'a')

    return min_dist, points_global