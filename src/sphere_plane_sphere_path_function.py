import numpy as np
import math
import copy
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed

# Including the following command to ensure that python is able to find the relevant files afer changing directory
sys.path.insert(0, '')
# Obtaining the current directory
cwd = os.getcwd()
current_directory = Path(__file__).parent
path_str = str(current_directory)

os.chdir(path_str)
from sphere_cylinder_sphere_function import config_sphere

# Importing code for the sphere
rel_path = '\Sphere code'
os.chdir(path_str + rel_path)
from Path_generation_sphere import optimal_path_sphere, generate_points_sphere

# Importing code for the plane
rel_path = '\Plane code'
os.chdir(path_str + rel_path)
from Plane_Dubins_functions import optimal_dubins_path

# Returning to initial directory
os.chdir(cwd)

def Path_generation_sphere_plane_sphere(ini_config, fin_config, center_ini_sphere, center_fin_sphere,\
                                           r, R_yaw, R_pitch, axis_plane, ht_plane, disc_no_loc, disc_no_heading, plot_figure_configs,\
                                           visualization = 1, filename = "temp.html", type = 'outer', vis_int = 0):
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
        Radius of the tight turn on sphere.
    R_pitch, R_yaw : Scalar
        Radius of the sphere corresponding to the maximum pitch and yaw rates.
    axis_plane : Array
        Axis of the line connecting the centers of the two spheres.
    ht_plane : Scalar
        Length of the plane connecting the considered pair of spheres.
    disc_no_loc, disc_no_heading : Scalar
        Number of discretizations in theta and phi considered, where theta represents
        the angle that dictates the plane that is chosen, and phi represents heading angle
        at the base and top of the plane.
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
    min_dist : Scalar
        Length of the shortest path connecting the two chosen spheres using a cross-tangent plane.
    points_global : Numpy array
        Contains the points along the shortest path connecting the two chosen spheres using a cross-tangent plane.
    tang_global : Numpy array
        Contains the direction cosines of the tangent vector in the global frame of the shortest path
        connecting the two chosen spheres using a cross-tangent plane.
    tang_normal_global : Numpy array
        Contains the direction cosines of the tangent normal vector in the global frame of the shortest path
        connecting the two chosen spheres using a cross-tangent plane.
    surf_normal_global : Numpy array
        Contains the direction cosines of the surface normal vector in the global frame of the shortest path
        connecting the two chosen spheres using a cross-tangent plane.
    
    '''

    # We first check if the considered connections exist
    # For the surface normal, depending on whether the first sphere is inner or outer, the sign will differ
    # if type == 'outer': sign = 1
    # else: sign = -1
    if type == 'outer': sign = 1; lrsign = 0 # "sign" is opposite to what we chose in the paper.
    # However, we account for this change in the code.
    elif type == 'inner': sign = -1; lrsign = 0
    elif type == 'left': lrsign = 1; sign = 0
    else: lrsign = -1; sign = 0

    # We compute the radius of the sphere
    R = abs(sign)*R_pitch + abs(lrsign)*R_yaw

    # We compute the radius of turn on the plane
    r_plane = (1 - abs(sign))*R_pitch + (1 - abs(lrsign))*R_yaw

    if 2*R > ht_plane: # In this case, the path does not exist
        return np.NaN, [], [], [], []

    # Discretizing the initial and final angles.
    # NOTE: thetai are generated such that they are in the interval
    # [0, 2pi). Therefore, they cannot take the value of 2pi, as this will then
    # cause a redundancy.
    thetai = np.linspace(0, 2*math.pi, disc_no_loc, endpoint = False)
    phii = np.linspace(-math.pi/2, math.pi/2, disc_no_heading)
    phio = np.linspace(-math.pi/2, math.pi/2, disc_no_heading)

    # We generate a random vector and orthonormalize it with respect to the axis
    # of the cylinder to obtain the x-axis using which theta is defined.
    flag = 0; counter = 0
    tol = 10**(-2) # tolerance for the dot product
    
    # We consider the x, y, and z vectors and consider to orthonoramlize them
    vect_arr = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    while flag == 0:

        vector = vect_arr[counter]

        if np.linalg.norm(-np.dot(vector, axis_plane)*axis_plane + vector) > tol:

            # In this case, we have obtained the desired x-axis for the body frame
            x = (-np.dot(vector, axis_plane)*axis_plane + vector)\
                /np.linalg.norm(-np.dot(vector, axis_plane)*axis_plane + vector)
            
            flag = 1

        else:

            if counter < 2:
                counter += 1
            else:
                raise Exception('Going into an infinite loop for generating the random vector')

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
                                    'Visualization of ' + type + ' sphere at initial configuration' +\
                                    ' and using cross-tangent plane to arrive at sphere at final configuration')
        
        # Writing the figure on the html file
        plot_figure.writing_fig_to_html(filename, 'a')

    # Final array to store path lengths for all configurations
    path_lengths = np.zeros((len(thetai), len(phii), len(phio)))

    # Storing the configuration corresponding to the minimum path length
    min_dist = np.infty; thetai_min = 0; phii_min = 0; phio_min = 0

    # Parameter for describing configuration on initial and final spheres
    alpha = math.acos(2*R/ht_plane)

    # We generate the y axis, centered at the center of the sphere at the initial configuration
    y = np.cross(axis_plane, x)

    # Now, we run the tasks in parallel for path length computation on the plane and the spheres
    ini_sphere_results = Parallel(n_jobs=-1, prefer="processes")(delayed(compute_path_ini_sphere)(i, j, thetai[i], phii[j], R, center_ini_sphere,\
                                                                                                  center_fin_sphere, r, vis_int, ini_config[0, :],\
                                                                                                  ini_config[1, :], alpha, axis_plane, x, y)\
                                                                                                  for i in range(len(thetai)) for j in range(len(phii)))
    plane_results = Parallel(n_jobs=-1, prefer="processes")(delayed(compute_path_plane)(i, j, phii[i], phio[j], ht_plane, r_plane, R, vis_int)\
                                                             for i in range(len(phii)) for j in range(len(phio)))
    fin_sphere_results = Parallel(n_jobs=-1, prefer="processes")(delayed(compute_path_fin_sphere)(i, j, thetai[i], phio[j], R, center_ini_sphere,\
                                                                                                center_fin_sphere, r, vis_int, fin_config[0, :],\
                                                                                                fin_config[1, :], alpha, axis_plane, x, y)\
                                                                                                for i in range(len(thetai)) for j in range(len(phio)))

    thetai_min = np.nan; phii_min = np.nan; phio_min = np.nan

    for res in ini_sphere_results:
        i, j = res['indices']
        path_lengths[i, j, :] += res['length']

    for res in plane_results:
        i, j = res['indices']
        path_lengths[:, i, j] += res['length']

    for res in fin_sphere_results:
        i, j = res['indices']
        path_lengths[i, :, j] += res['length']

    for i in range(len(thetai)):
        for j in range(len(phii)):
            for k in range(len(phio)):

                # path_lengths[i, j, k] = sp_1_path_lengths[i, j] + plane_path_lengths[j, k] + sp_2_path_lengths[i, k]
                if path_lengths[i, j, k] < min_dist:
                    
                    min_dist = path_lengths[i, j, k]
                    thetai_min = thetai[i]; phii_min = phii[j]; phio_min = phio[k]

    # We plot the optimal path
    # Obtaining the configuration for the spheres for exit from the first sphere and entry at final sphere
    loc_ini = center_ini_sphere + R*math.cos(alpha)*axis_plane +\
         R*math.sin(alpha)*(math.cos(thetai_min)*x + math.sin(thetai_min)*y)
    # Obtaining the location corresponding to the exit sphere
    loc_fin = center_fin_sphere - R*math.cos(alpha)*axis_plane + R*math.sin(alpha)*(math.cos(thetai_min + math.pi)*x \
            + math.sin(thetai_min + math.pi)*y)
    t = (loc_fin - loc_ini)/np.linalg.norm(loc_fin - loc_ini)
    # Obtaining the tangent vectors for the two spheres
    y_axis = np.cross((loc_ini - center_ini_sphere)/R, t)
    T_ini = math.cos(phii_min)*t + math.sin(phii_min)*(y_axis)
    T_fin = math.cos(phio_min)*t + math.sin(phio_min)*(y_axis)

    # Obtaining the optimal path on the first sphere
    ini_config_sphere = config_sphere(ini_config[0, :], center_ini_sphere, ini_config[1, :])
    fin_config_sphere = config_sphere(loc_ini, center_ini_sphere, T_ini)
    # Obtaining the best feasible path's portion on the first sphere
    _, _, _, minlen_sp1_path_points_x, minlen_sp1_path_points_y, minlen_sp1_path_points_z, minlen_sp1_Tx, minlen_sp1_Ty, minlen_sp1_Tz =\
        optimal_path_sphere(ini_config_sphere, fin_config_sphere, r, R, vis_int, path_config = 1, filename = "sp1_optimal_cross_tangent.html")[:9]
    
    # Obtaining the optimal path on the final sphere
    ini_config_sphere = config_sphere(loc_fin, center_fin_sphere, T_fin)
    fin_config_sphere = config_sphere(fin_config[0, :], center_fin_sphere, fin_config[1, :])
    # Obtaining the best feasible path's portion on the second sphere
    _, _, _, minlen_sp2_path_points_x, minlen_sp2_path_points_y, minlen_sp2_path_points_z, minlen_sp2_Tx, minlen_sp2_Ty, minlen_sp2_Tz =\
        optimal_path_sphere(ini_config_sphere, fin_config_sphere, r, R, vis_int, path_config = 1, filename = "sp2_optimal_cross_tangent.html")[:9]

    # Obtaining the optimal path on the plane
    ini_config_plane = np.array([0, 0, phii_min])
    fin_config_plane = np.array([math.sqrt(ht_plane**2 - 4*R**2), 0, phio_min])
    if vis_int == 1: filename_plane = 'optimal_path_cross_tangent_plane.html'
    else: filename_plane = False
    _, _, _, pts_x, pts_y, heading_opt = optimal_dubins_path(ini_config_plane, fin_config_plane, r, path_config = 1, filename = filename_plane)

    # Finding the global points of the path on the first sphere using a coordinate transformation
    # points_global_sp1 = np.empty((len(minlen_sp1_path_points_x), 3))
    points_global = np.empty((len(minlen_sp1_path_points_x) + len(pts_x) + len(minlen_sp2_path_points_x), 3))
    tang_global = np.empty((len(minlen_sp1_path_points_x) + len(pts_x) + len(minlen_sp2_path_points_x), 3))
    tang_normal_global = np.empty((len(minlen_sp1_path_points_x) + len(pts_x) + len(minlen_sp2_path_points_x), 3))
    surf_normal_global = np.empty((len(minlen_sp1_path_points_x) + len(pts_x) + len(minlen_sp2_path_points_x), 3))

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

    # Finding the global points of the path on the cross-tangent plane
    # points_global_plane = np.empty((len(pts_x), 3))
    sp1_pts_length = len(minlen_sp1_path_points_x)
    # We construct the rotation matrix relating the frame corresponding to the plane.
    rot_mat = np.array([[t[0], y_axis[0], ((loc_ini - center_ini_sphere)/R)[0]],\
                        [t[1], y_axis[1], ((loc_ini - center_ini_sphere)/R)[1]],\
                        [t[2], y_axis[2], ((loc_ini - center_ini_sphere)/R)[2]]])
    
    for i in range(len(pts_x)):

        ind = i + sp1_pts_length
        pos = loc_ini + np.matmul(rot_mat, np.array([pts_x[i], pts_y[i], 0]))
        points_global[ind, 0] = pos[0]
        points_global[ind, 1] = pos[1]
        points_global[ind, 2] = pos[2]

        # We obtain the tangent vector
        tang_vect = math.cos(heading_opt[i])*t + math.sin(heading_opt[i])*(y_axis)

        tang_global[ind, 0] = tang_vect[0]
        tang_global[ind, 1] = tang_vect[1]
        tang_global[ind, 2] = tang_vect[2]

        if sign != 0:
            # We obtain the tangent normal vector
            tang_norm = sign*np.cross((loc_ini - center_ini_sphere)/R, tang_global[ind])
            tang_normal_global[ind, 0] = tang_norm[0]
            tang_normal_global[ind, 1] = tang_norm[1]
            tang_normal_global[ind, 2] = tang_norm[2]

            # The surface normal remains the same as that on the sphere when it exits
            surf_normal_global[ind, 0] = (sign*(loc_ini - center_ini_sphere)/R)[0]
            surf_normal_global[ind, 1] = (sign*(loc_ini - center_ini_sphere)/R)[1]
            surf_normal_global[ind, 2] = (sign*(loc_ini - center_ini_sphere)/R)[2]

        else:

            tang_norm = -lrsign*(loc_ini - center_ini_sphere)/R
            tang_normal_global[ind, 0] = tang_norm[0]
            tang_normal_global[ind, 1] = tang_norm[1]
            tang_normal_global[ind, 2] = tang_norm[2]

            # Obtaining the surface normal
            surf_norm = np.cross(tang_global[ind], tang_normal_global[ind])
            surf_normal_global[ind, 0] = surf_norm[0]
            surf_normal_global[ind, 1] = surf_norm[1]
            surf_normal_global[ind, 2] = surf_norm[2]

    # Finding the global points of the path on the last sphere using a coordinate transformation
    # points_global_sp2 = np.empty((len(minlen_sp2_path_points_x), 3))
    sp1_pls_plane_points_length = len(pts_x) + sp1_pts_length
    for i in range(len(minlen_sp2_path_points_x)):

        ind = i + sp1_pls_plane_points_length
        points_global[ind, 0] = minlen_sp2_path_points_x[i] + center_fin_sphere[0]
        points_global[ind, 1] = minlen_sp2_path_points_y[i] + center_fin_sphere[1]
        points_global[ind, 2] = minlen_sp2_path_points_z[i] + center_fin_sphere[2]
        tang_global[ind, 0] = minlen_sp2_Tx[i]; tang_global[ind, 1] = minlen_sp2_Ty[i]; tang_global[ind, 2] = minlen_sp2_Tz[i]
        
        if sign != 0: # In this case, inner and outer spheres have been considered

            # Sign opposite to initial sphere is considered
            surf_normal_global[ind, 0] = -sign*minlen_sp2_path_points_x[i]/R; 
            surf_normal_global[ind, 1] = -sign*minlen_sp2_path_points_y[i]/R; 
            surf_normal_global[ind, 2] = -sign*minlen_sp2_path_points_z[i]/R; 
            tang_normal_global[ind, :] = np.cross(surf_normal_global[ind], tang_global[ind])

        else:

            tang_normal_global[ind, 0] = lrsign*minlen_sp2_path_points_x[i]/R; 
            tang_normal_global[ind, 1] = lrsign*minlen_sp2_path_points_y[i]/R; 
            tang_normal_global[ind, 2] = lrsign*minlen_sp2_path_points_z[i]/R; 
            surf_normal_global[ind, :] = np.cross(tang_global[ind], tang_normal_global[ind])

    if visualization == 1:

        # Plotting the path
        plot_figure.scatter_3D(points_global[:, 0], points_global[:, 1],\
                                points_global[:, 2], 'blue', 'Optimal path')  

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
                                    'Best feasible path connecting ' + type + ' sphere at initial configuration using a cross-tangent plane')

        # Writing the figure on the html file
        plot_figure.writing_fig_to_html(filename, 'a')

    return min_dist, points_global, tang_global, tang_normal_global, surf_normal_global

def compute_path_plane(i, j, phii, phio, ht_plane, r_plane, R, vis_int):

    result = {}

    if vis_int == 0: filename_plane = False
    else: filename_plane = "plane_phii_" + str(i) + "_phio_" + str(j) + ".html"

    # We obtain the initial and final configuration on the plane
    ini_config_plane = np.array([0, 0, phii])
    fin_config_plane = np.array([math.sqrt(ht_plane**2 - 4*R**2), 0, phio])
    # Note that ht_plane is the distance between the center of the initial and final spheres

    result['indices'] = (i, j)
    result['length'] = optimal_dubins_path(ini_config_plane, fin_config_plane, r_plane, path_config = vis_int, filename = filename_plane)[0]

    return result

def compute_path_ini_sphere(i, j, theta, phi, R, center_ini_sphere, center_fin_sphere, r, vis_int, ini_loc, ini_tang, alpha, axis_plane, x, y):

    result = {}

    # Obtaining the configuration for the sphere
    loc_ini = center_ini_sphere + R*math.cos(alpha)*axis_plane + R*math.sin(alpha)*(math.cos(theta)*x + math.sin(theta)*y)
    # Obtaining the location corresponding to the exit sphere
    loc_fin = center_fin_sphere - R*math.cos(alpha)*axis_plane + R*math.sin(alpha)*(math.cos(theta + math.pi)*x \
            + math.sin(theta + math.pi)*y)

    # Obtaining the tangent vector
    t = (loc_fin - loc_ini)/np.linalg.norm(loc_fin - loc_ini)

    # Tangent vector for exit from sphere at initial configuration
    T_ini = math.cos(phi)*t + math.sin(phi)*(np.cross((loc_ini - center_ini_sphere)/R, t))
    
    # We now obtain the initial and final configurations for the chosen sphere at the initial configuration
    ini_sphere_ini_config = config_sphere(ini_loc, center_ini_sphere, ini_tang)
    ini_sphere_fin_config = config_sphere(loc_ini, center_ini_sphere, T_ini)

    # print('Radius of sphere is ', R, ' and norm of distance is ', np.linalg.norm(ini_sphere_fin_config[:, 0]))
    # print('Initial sphere ini config is ', ini_sphere_ini_config, ' and final config is ', ini_sphere_fin_config)
    
    filename_sp = "sp_1_thetai_" + str(i) + "_phii_" + str(j) + ".html"
    result['indices'] = (i, j)
    result['length'] =\
            optimal_path_sphere(ini_sphere_ini_config, ini_sphere_fin_config, r, R, vis_int, path_config = vis_int, filename = filename_sp)[1]

    return result

def compute_path_fin_sphere(i, j, theta, phi, R, center_ini_sphere, center_fin_sphere, r, vis_int, fin_loc, fin_tang, alpha, axis_plane, x, y):

    result = {}

    # Obtaining the configuration for the sphere
    loc_ini = center_ini_sphere + R*math.cos(alpha)*axis_plane + R*math.sin(alpha)*(math.cos(theta)*x + math.sin(theta)*y)
    # Obtaining the location corresponding to the exit sphere
    loc_fin = center_fin_sphere - R*math.cos(alpha)*axis_plane + R*math.sin(alpha)*(math.cos(theta + math.pi)*x \
            + math.sin(theta + math.pi)*y)

    # Obtaining the tangent vector
    t = (loc_fin - loc_ini)/np.linalg.norm(loc_fin - loc_ini)

    # Tangent vector for entry onto sphere at final configuration
    T_fin = math.cos(phi)*t + math.sin(phi)*(np.cross((loc_ini - center_ini_sphere)/R, t))

    # Now, we construct the configuration for planning on the final sphere
    fin_sphere_ini_config = config_sphere(loc_fin, center_fin_sphere, T_fin)

    # Now, we construct the configuration for exit on the final sphere
    fin_sphere_fin_config = config_sphere(fin_loc, center_fin_sphere, fin_tang)

    filename_sp = "sp_2_thetai_" + str(i) + "_phio_" + str(j) + ".html"
    result['indices'] = (i, j)
    result['length'] =\
            optimal_path_sphere(fin_sphere_ini_config, fin_sphere_fin_config, r, R, vis_int, path_config = vis_int, filename = filename_sp)[1]
    
    return result