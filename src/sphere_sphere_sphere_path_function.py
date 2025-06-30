import numpy as np
import math
import copy
import os
import sys
from math import cos as cos
from math import sin as sin
from math import sqrt as sqrt
from pathlib import Path
from joblib import Parallel, delayed

# Including the following command to ensure that python is able to find the relevant files afer changing directory
sys.path.insert(0, '')
# Obtaining the current directory
cwd = os.getcwd()
current_directory = Path(__file__).parent
path_str = str(current_directory)

from sphere_cylinder_sphere_function_old import config_sphere

# Importing code for the sphere
rel_path = '\Sphere code'
os.chdir(path_str + rel_path)
from Path_generation_sphere import optimal_path_sphere, generate_points_sphere

# Returning to initial directory
os.chdir(cwd)

def Path_generation_sphere_sphere_sphere(ini_config, fin_config, center_ini_sphere, center_fin_sphere,\
                                           r, R, axis_plane, dist_center_spheres, disc_no_loc, disc_no_heading, plot_figure_configs,\
                                           visualization = 1, filename = "temp.html", type = 'outer', vis_int = 0):
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
    min_dist : Scalar
        Length of the shortest path connecting the two chosen spheres using an intermediary sphere.
    points_global : Numpy array
        Contains the points along the shortest path connecting the two chosen spheres using an intermediary sphere.
    tang_global : Numpy array
        Contains the direction cosines of the tangent vector in the global frame of the shortest path
        connecting the two chosen spheres using an intermediary sphere.
    tang_normal_global : Numpy array
        Contains the direction cosines of the tangent normal vector in the global frame of the shortest path
        connecting the two chosen spheres using an intermediary sphere.
    surf_normal_global : Numpy array
        Contains the direction cosines of the surface normal vector in the global frame of the shortest path
        connecting the two chosen spheres using an intermediary sphere.

    '''

    # We first check for existence of such a connection depending on the distance between
    # the centers of the spheres
    if dist_center_spheres > 4*R:
        return np.NaN, [], [], [], []

    # Discretizing the angle for parameterizing the intermediary sphere and the
    # angles for the tangent vector for exit from initial sphere and entry into final
    # sphere
    theta = np.linspace(0, 2*math.pi, disc_no_loc, endpoint = False)
    phiic = np.linspace(0, 2*math.pi, disc_no_heading, endpoint = False)
    phifc = np.linspace(0, 2*math.pi, disc_no_heading, endpoint = False)

    # Obtaining the angle to describe the locus of the intermediary sphere
    alpha = math.acos(dist_center_spheres/(4*R))

    # Generating a random vector x perpendicular to the vector connecting the centers of the
    # initial and final spheres.
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
                                    ' via sphere connecting ' + type + ' spheres')
        
        # Writing the figure on the html file
        plot_figure.writing_fig_to_html(filename, 'a')
        
    path_lengths = np.zeros((len(theta), len(phiic), len(phifc)))

    # We run through each discretization
    y = np.cross(axis_plane, x)
    
    ini_sphere_results = Parallel(n_jobs=-1, prefer="processes")(delayed(compute_path_ini_sphere)(i, j, theta[i], phiic[j], alpha, x, y,\
                                                                                                R, center_ini_sphere, center_fin_sphere, r,\
                                                                                                vis_int, ini_config[0, :], ini_config[1, :], axis_plane)\
                                                                                                for i in range(len(theta)) for j in range(len(phiic)))
    int_sphere_results = Parallel(n_jobs=-1, prefer="processes")(delayed(compute_path_int_sphere)(i, j, phiic[i], phifc[j], theta[0], alpha, x, y,\
                                                                                                R, center_ini_sphere, center_fin_sphere, r,\
                                                                                                vis_int, axis_plane)\
                                                                                                for i in range(len(phiic)) for j in range(len(phifc)))
    fin_sphere_results = Parallel(n_jobs=-1, prefer="processes")(delayed(compute_path_fin_sphere)(i, j, theta[i], phifc[j], alpha, x, y,\
                                                                                                R, center_ini_sphere, center_fin_sphere, r,\
                                                                                                vis_int, fin_config[0, :], fin_config[1, :], axis_plane)\
                                                                                                for i in range(len(theta)) for j in range(len(phifc)))
                
    # Process results
    for res in ini_sphere_results:
        i, j = res['indices']
        path_lengths[i, j, :] += res['length']

    for res in int_sphere_results:
        i, j = res['indices']
        path_lengths[:, i, j] += res['length']

    for res in fin_sphere_results:
        i, j = res['indices']
        path_lengths[i, :, j] += res['length']

    # We now pick the best path
    min_dist = np.inf
    theta_min = 0; phiic_min = 0; phifc_min = 0
    for i in range(len(theta)):
        for j in range(len(phiic)):
            for k in range(len(phifc)):

                # path_lengths[i, j, k] = sp1_path_lengths[i, j] + sp2_path_lengths[i, k] + spint_path_lengths[i, j, k]
                if path_lengths[i, j, k] < min_dist:

                    min_dist = path_lengths[i, j, k]
                    theta_min = theta[i]; phiic_min = phiic[j]; phifc_min = phifc[k]

    # We obtain the points along the optimal path and plot it if needed
    # Obtaining the configuration for the spheres for exit from the first sphere and entry at final sphere
    Xcthetac = center_ini_sphere + 0.5*(center_fin_sphere - center_ini_sphere)\
            + 2*R*sin(alpha)*(cos(theta_min)*x + sin(theta_min)*y)
    # Now, we obtain the location corresponding to exit from the initial sphere
    # and entry onto the final sphere
    Xic = 0.5*(center_ini_sphere + Xcthetac)
    Xoc = 0.5*(center_fin_sphere + Xcthetac)

    # We now obtain the vectors for parameterizing the tangent vector
    xic = axis_plane/sin(alpha) - (Xic - center_ini_sphere)/(R*math.tan(alpha))
    xoc = -axis_plane/sin(alpha) - (Xoc - center_fin_sphere)/(R*math.tan(alpha))

    # print('Norm of xic and xoc are ', np.linalg.norm(xic), np.linalg.norm(xoc))
    # print('Dot product of xic and xoc with the position is ', np.dot(xic, Xic - center_ini_sphere), np.dot(xoc, Xoc - center_fin_sphere))

    # We obtain the tangent vector at the initial sphere
    Tic = cos(phiic_min)*xic + sin(phiic_min)*(1/R)*np.cross(Xic - center_ini_sphere, xic)
    # We obtain the tangent vector at the final sphere
    Toc = cos(phifc_min)*xoc + sin(phifc_min)*(1/R)*np.cross(Xoc - center_fin_sphere, xoc)

    # Obtaining the optimal path on all three spheres
    # Initial sphere
    ini_config_sp1 = config_sphere(ini_config[0, :], center_ini_sphere, ini_config[1, :])
    fin_config_sp1 = config_sphere(Xic, center_ini_sphere, Tic)
    filename_sp = "sp_1_optimal" + ".html"
    _, _, _, minlen_sp1_path_points_x, minlen_sp1_path_points_y, minlen_sp1_path_points_z, minlen_sp1_Tx, minlen_sp1_Ty, minlen_sp1_Tz =\
            optimal_path_sphere(ini_config_sp1, fin_config_sp1, r, R, vis_int, path_config = 1, filename = filename_sp)[0:9]
    
    # Intermediary sphere
    ini_config_spint = config_sphere(Xic, Xcthetac, Tic)
    fin_config_spint = config_sphere(Xoc, Xcthetac, Toc)
    filename_sp = "sp_int_optimal" + ".html"

    _, _, _, minlen_spint_path_points_x, minlen_spint_path_points_y, minlen_spint_path_points_z, minlen_spint_Tx, minlen_spint_Ty, minlen_spint_Tz =\
            optimal_path_sphere(ini_config_spint, fin_config_spint, r, R, vis_int, path_config = 1, filename = filename_sp)[0:9]
    
    # Final sphere
    ini_config_sp2 = config_sphere(Xoc, center_fin_sphere, Toc)
    fin_config_sp2 = config_sphere(fin_config[0, :], center_fin_sphere, fin_config[1, :])
    filename_sp = "sp_2_optimal" + ".html"

    _, _, _, minlen_sp2_path_points_x, minlen_sp2_path_points_y, minlen_sp2_path_points_z, minlen_sp2_Tx, minlen_sp2_Ty, minlen_sp2_Tz =\
            optimal_path_sphere(ini_config_sp2, fin_config_sp2, r, R, vis_int, path_config = 1, filename = filename_sp)[0:9]
    
    # Finding the global points of the path on the first sphere using a coordinate transformation
    points_global = np.empty((len(minlen_sp1_path_points_x) + len(minlen_spint_path_points_x) + len(minlen_sp2_path_points_x), 3))
    tang_global = np.empty((len(minlen_sp1_path_points_x) + len(minlen_spint_path_points_x) + len(minlen_sp2_path_points_x), 3))
    tang_normal_global = np.empty((len(minlen_sp1_path_points_x) + len(minlen_spint_path_points_x) + len(minlen_sp2_path_points_x), 3))
    surf_normal_global = np.empty((len(minlen_sp1_path_points_x) + len(minlen_spint_path_points_x) + len(minlen_sp2_path_points_x), 3))

    if type == 'outer': sign = 1; lrsign = 0
    elif type == 'inner': sign = -1; lrsign = 0
    elif type == 'left': lrsign = 1; sign = 0
    else: lrsign = -1; sign = 0

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

    # Finding the global points of the path on the intermediary sphere
    sp1_pts_length = len(minlen_sp1_path_points_x)

    for i in range(len(minlen_spint_path_points_x)):

        ind = sp1_pts_length + i
        points_global[ind, 0] = minlen_spint_path_points_x[i] + Xcthetac[0]
        points_global[ind, 1] = minlen_spint_path_points_y[i] + Xcthetac[1]
        points_global[ind, 2] = minlen_spint_path_points_z[i] + Xcthetac[2]
        tang_global[ind, 0] = minlen_spint_Tx[i]; tang_global[ind, 1] = minlen_spint_Ty[i]; tang_global[ind, 2] = minlen_spint_Tz[i]

        if sign != 0: # In this case, inner and outer spheres have been considered. NOTE: We need to flip the sign for intermediary
            # sphere

            surf_normal_global[ind, 0] = -sign*minlen_spint_path_points_x[i]/R; 
            surf_normal_global[ind, 1] = -sign*minlen_spint_path_points_y[i]/R; 
            surf_normal_global[ind, 2] = -sign*minlen_spint_path_points_z[i]/R; 
            tang_normal_global[ind, :] = np.cross(surf_normal_global[ind], tang_global[ind])

        else:

            tang_normal_global[ind, 0] = lrsign*minlen_spint_path_points_x[i]/R; 
            tang_normal_global[ind, 1] = lrsign*minlen_spint_path_points_y[i]/R; 
            tang_normal_global[ind, 2] = lrsign*minlen_spint_path_points_z[i]/R; 
            surf_normal_global[ind, :] = np.cross(tang_global[ind], tang_normal_global[ind])

    # Finding global points of the path on the final sphere
    sp1_plus_int_pts_length = sp1_pts_length + len(minlen_spint_path_points_x)

    for i in range(len(minlen_sp2_path_points_x)):

        ind = sp1_plus_int_pts_length + i
        points_global[ind, 0] = minlen_sp2_path_points_x[i] + center_fin_sphere[0]
        points_global[ind, 1] = minlen_sp2_path_points_y[i] + center_fin_sphere[1]
        points_global[ind, 2] = minlen_sp2_path_points_z[i] + center_fin_sphere[2]
        tang_global[ind, 0] = minlen_sp2_Tx[i]; tang_global[ind, 1] = minlen_sp2_Ty[i]; tang_global[ind, 2] = minlen_sp2_Tz[i]

        if sign != 0: # In this case, inner and outer spheres have been considered.

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

        # Plotting the path on the first sphere
        plot_figure.scatter_3D(points_global[:, 0], points_global[:, 1],\
                                points_global[:, 2], 'blue', 'Optimal path')

        # Plotting the intermediary sphere as well
        plot_figure.surface_3D(generate_points_sphere(Xcthetac, R)[0],\
                               generate_points_sphere(Xcthetac, R)[1],\
                               generate_points_sphere(Xcthetac, R)[2], 'grey',\
                               'Final sphere', 0.7)
        
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
                                    'Best feasible path connecting ' + type + ' spheres using an intermediary sphere')

        # Writing the figure on the html file
        plot_figure.writing_fig_to_html(filename, 'a')

    return min_dist, points_global, tang_global, tang_normal_global, surf_normal_global

def compute_path_ini_sphere(i, j, theta, phii, alpha, x, y, R, center_ini_sphere, center_fin_sphere, r, vis_int, ini_loc, ini_tang, axis_plane):

    result = {}

    # We now obtain the expression for the center of the intermediary sphere
    Xcthetac = center_ini_sphere + 0.5*(center_fin_sphere - center_ini_sphere)\
        + 2*R*sin(alpha)*(cos(theta)*x + sin(theta)*y)
    
    # print('Distance of center of sphere from initial sphere is ', np.linalg.norm(Xcthetac - center_ini_sphere))
    # print('Distance of center of sphere from final sphere is ', np.linalg.norm(Xcthetac - center_fin_sphere))
    
    # Now, we obtain the location corresponding to exit from the initial sphere
    # and entry onto the final sphere
    Xic = 0.5*(center_ini_sphere + Xcthetac)

    # We now obtain the vectors for parameterizing the tangent vector
    xic = axis_plane/sin(alpha) - (Xic - center_ini_sphere)/(R*math.tan(alpha))

    # We obtain the tangent vector at the initial sphere
    Tic = cos(phii)*xic + sin(phii)*(1/R)*np.cross(Xic - center_ini_sphere, xic)

    # We compute the path on the initial sphere
    ini_config_sp1 = config_sphere(ini_loc, center_ini_sphere, ini_tang)
    fin_config_sp1 = config_sphere(Xic, center_ini_sphere, Tic)
    filename_sp = "sp_1_thetai_" + str(i) + "_phiic_" + str(j) + ".html"
    
    result['indices'] = (i, j)
    result['length'] =\
            optimal_path_sphere(ini_config_sp1, fin_config_sp1, r, R, vis_int, path_config = vis_int, filename = filename_sp)[1]

    return result

def compute_path_fin_sphere(i, j, theta, phio, alpha, x, y, R, center_ini_sphere, center_fin_sphere, r, vis_int, fin_loc, fin_tang, axis_plane):

    result = {}

    # We now obtain the expression for the center of the intermediary sphere
    Xcthetac = center_ini_sphere + 0.5*(center_fin_sphere - center_ini_sphere)\
        + 2*R*sin(alpha)*(cos(theta)*x + sin(theta)*y)
    
    # print('Distance of center of sphere from initial sphere is ', np.linalg.norm(Xcthetac - center_ini_sphere))
    # print('Distance of center of sphere from final sphere is ', np.linalg.norm(Xcthetac - center_fin_sphere))
    
    # Now, we obtain the location corresponding to exit from the initial sphere
    # and entry onto the final sphere
    Xoc = 0.5*(center_fin_sphere + Xcthetac)

    # We now obtain the vectors for parameterizing the tangent vector
    xoc = -axis_plane/sin(alpha) - (Xoc - center_fin_sphere)/(R*math.tan(alpha))

    # We obtain the tangent vector at the final sphere
    Toc = cos(phio)*xoc + sin(phio)*(1/R)*np.cross(Xoc - center_fin_sphere, xoc)

    # We compute the path on the initial sphere
    ini_config_sp2 = config_sphere(Xoc, center_fin_sphere, Toc)
    fin_config_sp2 = config_sphere(fin_loc, center_fin_sphere, fin_tang)
    filename_sp = "sp_2_thetai_" + str(i) + "_phifc_" + str(j) + ".html"
    
    result['indices'] = (i, j)
    result['length'] =\
            optimal_path_sphere(ini_config_sp2, fin_config_sp2, r, R, vis_int, path_config = vis_int, filename = filename_sp)[1]

    return result

def compute_path_int_sphere(i, j, phii, phio, theta, alpha, x, y, R, center_ini_sphere, center_fin_sphere, r, vis_int, axis_plane):
    # NOTE: This computation suffices to be performed for one theta value; as the intermediary sphere rotates, the entry and exit
    # configurations rotates correspondingly. When we set the initial configuration to be the identity matrix, we get the same final configuration
    # on the intermediary sphere for all theta values.

    result = {}

    # We now obtain the expression for the center of the intermediary sphere
    Xcthetac = center_ini_sphere + 0.5*(center_fin_sphere - center_ini_sphere)\
        + 2*R*sin(alpha)*(cos(theta)*x + sin(theta)*y)
    
    # print('Distance of center of sphere from initial sphere is ', np.linalg.norm(Xcthetac - center_ini_sphere))
    # print('Distance of center of sphere from final sphere is ', np.linalg.norm(Xcthetac - center_fin_sphere))
    
    # Now, we obtain the location corresponding to exit from the initial sphere
    # and entry onto the final sphere
    Xic = 0.5*(center_ini_sphere + Xcthetac)
    Xoc = 0.5*(center_fin_sphere + Xcthetac)

    # We now obtain the vectors for parameterizing the tangent vector
    xic = axis_plane/sin(alpha) - (Xic - center_ini_sphere)/(R*math.tan(alpha))
    xoc = -axis_plane/sin(alpha) - (Xoc - center_fin_sphere)/(R*math.tan(alpha))

    # We obtain the tangent vector at the initial and final spheres
    Tic = cos(phii)*xic + sin(phii)*(1/R)*np.cross(Xic - center_ini_sphere, xic)
    Toc = cos(phio)*xoc + sin(phio)*(1/R)*np.cross(Xoc - center_fin_sphere, xoc)

    # We compute the path on the initial sphere
    ini_config_spint = config_sphere(Xic, Xcthetac, Tic)
    fin_config_spint = config_sphere(Xoc, Xcthetac, Toc)
    filename_sp = "sp_int_phiic_" + str(i) + "_phifc_" + str(j) + ".html"
    
    result['indices'] = (i, j)
    result['length'] =\
            optimal_path_sphere(ini_config_spint, fin_config_spint, r, R, vis_int, path_config = vis_int, filename = filename_sp)[1]

    return result