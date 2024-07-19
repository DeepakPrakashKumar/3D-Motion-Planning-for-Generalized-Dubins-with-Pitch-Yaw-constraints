# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 17:25:54 2022

@author: deepa
"""

import numpy as np
import math
from math import cos as cos
from math import sin as sin
from math import sqrt as sqrt
import os
import copy
import numpy.polynomial.polynomial as poly
import sys

# Importing functions for generating the 3D plots
path = 'D:\TAMU\Research\Cylinder code'
# os.chdir(path)
sys.path.append(path)
# print(os.getcwd())
from plotting_class import plotting_functions

# Returning to original directory
path = 'D:\TAMU\Research\ThreeD Dubins sphere\Code path generation using inverse kinematics'
# os.chdir(path)
sys.path.append(path)
# print(os.getcwd())

def operators_segments(ini_config, phi, r, R, seg_type = 'l'):
    '''
    This function defines the operators corresponding to left, , and great
    circle turns on a sphere.

    Parameters
    ----------
    ini_config : Numpy array
        Contains the configuration before the segment. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    phi : Scalar
        Describes the angle of the turn.
    r : Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the sphere.
    seg_type : Character, optional
        Defines the type of segment considered.

    Returns
    -------
    fin_config : Numpy array
        Contains the configuration after the corresponding segment. The syntax
        followed is the same as the initial configuration.

    '''
        
    # Defining the scaled radius of the turn
    rb = r/R
    
    # Defining the matrix corresponding to each segment type given the angle
    # of the turn
    if seg_type.lower() == 'g':
        
        R = np.array([[math.cos(phi), -(1/R)*math.sin(phi), 0],\
                      [R*math.sin(phi), math.cos(phi), 0],\
                      [0, 0, 1]])
    
    elif seg_type.lower() == 'l':
        
        R = np.array([[1 - (1 - math.cos(phi))*rb**2, -(rb/R)*math.sin(phi),\
                       (1/R)*(1 - math.cos(phi))*rb*math.sqrt(1 - rb**2)],\
                      [r*math.sin(phi), math.cos(phi), -math.sin(phi)*math.sqrt(1 - rb**2)],\
                      [(1 - math.cos(phi))*r*math.sqrt(1 - rb**2),\
                       math.sin(phi)*math.sqrt(1 - rb**2),\
                       math.cos(phi) + (1 - math.cos(phi))*rb**2]])
            
    elif seg_type.lower() == 'r':
        
        R = np.array([[1 - (1 - math.cos(phi))*rb**2, -(rb/R)*math.sin(phi),\
                       -(1/R)*(1 - math.cos(phi))*rb*math.sqrt(1 - rb**2)],\
                      [r*math.sin(phi), math.cos(phi), math.sin(phi)*math.sqrt(1 - rb**2)],\
                      [-(1 - math.cos(phi))*r*math.sqrt(1 - rb**2),\
                       -math.sin(phi)*math.sqrt(1 - rb**2),\
                       math.cos(phi) + (1 - math.cos(phi))*rb**2]])
            
    # Obtaining the final configuration
    fin_config = np.matmul(ini_config, R)
    
    return fin_config

def Seg_pts(ini_config, phi, r, R, seg_type = 'l'):
    '''
    This function generates points along the segment on a sphere. Moreover, the
    tangent vector at the generated points are also returned.

    Parameters
    ----------
    ini_config : Numpy array
        Contains the configuration before the segment. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    phi : Scalar
        Describes the angle of the turn.
    r : Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the sphere.
    seg_type : Character, optional
        Defines the type of segment considered.

    Returns
    -------
    pos_array : Numpy array
        Contains the coordinates of points along the segment on a sphere.
    tang_array : Numpy array
        Contains the tangent vector at the generated points along the segment on a sphere.

    '''
    
    # Checking if a valid segment type is passed
    seg_types_allowed = ['l', 'r', 'g']
    
    if seg_type.lower() not in seg_types_allowed:
        
        raise Exception('Incorrect path type passed.')
        
    # Discretizing the angle of the turn
    phi_disc = np.linspace(0, phi, 100)
    
    # Declaring the arrays used to store the positions and the tangent vectors
    pos_array = np.empty((100, 3))
    tang_array = np.empty((100, 3))
    
    for i in range(len(phi_disc)):
        
        # Running the function in which the operators are defined
        config_turn = operators_segments(ini_config, phi_disc[i], r, R, seg_type)
        # Extracting the position and tangent vectors from config_turn
        pos_array[i, 0] = config_turn[0, 0]
        pos_array[i, 1] = config_turn[1, 0]
        pos_array[i, 2] = config_turn[2, 0]
        tang_array[i, 0] = config_turn[0, 1]
        tang_array[i, 1] = config_turn[1, 1]
        tang_array[i, 2] = config_turn[2, 1]
    
    return pos_array, tang_array

def points_path(ini_config, rL, rR, R, angle_segments, path_type = 'lgl'):
    '''
    This function generates points along a path on the sphere. Moreover, points
    along the circles corresponding to each segment of the path are also returned.

    Parameters
    ----------
    ini_config : Numpy array
        Contains the configuration before the segment. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    rL, rR : Scalar
        Radius of the left tight turn and  tight turn.
    R : Scalar
        Radius of the sphere.
    angle_segments : Numpy array
        Contains the angle for each segment of the path.
    path_type : String, optional
        Defines the type of path considered. The default is 'lgl'.

    Returns
    -------
    x_coords_path, y_coords_path, z_coords_path : Numpy arrays
        Contains the coordinates of points along the path.
    fin_config_path : Numpy array
        Contains the final configuration obtained after the path.
    x_coords_circles, y_coords_circles, z_coords_circles : Numpy arrays
        Contains the coordinates of points along the circles corresponding 
        to each segment of the path.

    '''
            
    if len(angle_segments) < len(path_type):
        
        raise Exception('Number of parameters of the path is lesser than the number '\
                        + 'of segments of the path.')
            
    # Declaring the position array to store coordinates of points along the path
    x_coords_path = []
    y_coords_path = []
    z_coords_path = []
    # Declaring the position array to store coordinates of points along the circles
    # corresponding to the path
    x_coords_circles = []
    y_coords_circles = []
    z_coords_circles = []
    
    # Storing the initial configuration before every segment
    ini_config_seg = ini_config
    
    for i in range(len(path_type)):
        
        # Obtaining the points along the path
        if path_type[i].lower() == 'l':
            
            r = rL
            
        else:
            
            r = rR
        
        points_path_seg, _ = Seg_pts(ini_config_seg, angle_segments[i], r, R, path_type[i])
        # Appending the obtained points to the arrays
        x_coords_path = np.append(x_coords_path, points_path_seg[:, 0])
        y_coords_path = np.append(y_coords_path, points_path_seg[:, 1])
        z_coords_path = np.append(z_coords_path, points_path_seg[:, 2])
        
        # Updating the initial configuration for the next segment to the final 
        # configuration of the considered segment of the path
        ini_config_seg = operators_segments(ini_config_seg, angle_segments[i], r, R, path_type[i]) 
        
        # Obtaining the points along the circle corresponding to the ith segment of the path
        # NOTE THAT THE POINTS ARE OBTAINED SUCH THAT THEY DO NOT OVERLAP WITH POINTS ON THE PATH
        points_path_seg, _ = Seg_pts(ini_config_seg, 2*math.pi - angle_segments[i],\
                                     r, R, path_type[i])
        x_coords_circles = np.append(x_coords_circles, points_path_seg[:, 0])
        y_coords_circles = np.append(y_coords_circles, points_path_seg[:, 1])
        z_coords_circles = np.append(z_coords_circles, points_path_seg[:, 2])
        
    fin_config_path = ini_config_seg
    
    return x_coords_path, y_coords_path, z_coords_path, fin_config_path, x_coords_circles,\
        y_coords_circles, z_coords_circles
        
def modifying_initial_final_configurations_unit_sphere(ini_config, fin_config, R):
    '''
    In this function, the initial and final configuration is modified for the purpose
    of inverse kinematics such that
        - the sphere is unit
        - the initial configuration is the identity matrix, i.e., the initial
        location is on the x-axis and the initial tangent vector is oriented along
        the y-axis.

    Parameters
    ----------
    ini_config : Numpy array
        Contains the initial configuration. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    fin_config : Numpy array
        Contains the final configuration. The same syntax used for ini_config is
        used here.
    R : Scalar
        Radius of the sphere.

    Returns
    -------
    fin_config_mod : Numpy array
        Contains the final configuration lying on a unit sphere such that the initial
        configuration matrix is the identity matrix.

    '''
    
    # Obtaining the new final configuration after scaling
    fin_config_scaled = np.array([[fin_config[0, 0]/R, fin_config[0, 1], fin_config[0, 2]],\
                                  [fin_config[1, 0]/R, fin_config[1, 1], fin_config[1, 2]],\
                                  [fin_config[2, 0]/R, fin_config[2, 1], fin_config[2, 2]]])
    # Obtaining the scaled initial configuration
    ini_config_scaled = np.array([[ini_config[0, 0]/R, ini_config[0, 1], ini_config[0, 2]],\
                                  [ini_config[1, 0]/R, ini_config[1, 1], ini_config[1, 2]],\
                                  [ini_config[2, 0]/R, ini_config[2, 1], ini_config[2, 2]]])
    # Modifying the final configuration such that the initial configuration coincides
    # with the identity matrix. NOTE: For any path, Rfinal = Rinitial x Rnetrotation. Hence,
    # Rfinal_mod = Rinitial**T x Rfinal
    fin_config_mod = np.matmul(ini_config_scaled.transpose(), fin_config_scaled)
    
    return fin_config_mod

def path_generation_sphere_GCG(ini_config, fin_config, r, R, mu, path_type = 'glg'):
    
    # Modifying the configurations and the parameters of the turn
    fin_config_mod = modifying_initial_final_configurations_unit_sphere(ini_config, fin_config, R)
    r_mod = r/R
    
    path_types = ['glg', 'grg']
    
    if path_type not in path_types:
        
        raise Exception('Incorrect path type passed.')
        
    # Defining variables corresponding to the final configuration
    alpha11 = fin_config_mod[0, 0]; alpha12 = fin_config_mod[0, 1]; alpha13 = fin_config_mod[0, 2];
    alpha21 = fin_config_mod[1, 0]; alpha22 = fin_config_mod[1, 1]; alpha23 = fin_config_mod[1, 2];
    alpha31 = fin_config_mod[2, 0]; alpha32 = fin_config_mod[2, 1]; alpha33 = fin_config_mod[2, 2];
    
    # Storing the details of the path
    path_params = []
        
    # Obtaining the solutions for phi2
    cphi2 = (alpha33 - r_mod**2)/(1 - r_mod**2)
        
    # Checking if solution exists. Also, account for numerical inaccuracies.
    if abs(cphi2) > 1 and abs(cphi2) <= 1 + 10**(-6):
        
        cphi2 = np.sign(cphi2)
        
    # Checking if path exists
    if abs(cphi2) > 1:
        
        print(path_type.upper() + ' path does not exist.')
        path_length = np.NaN; phi1 = np.NaN; phi2 = np.NaN; phi3 = np.NaN;
        path_params.append([path_length, phi1, phi2, phi3])
        
        return path_params
    
    # Checking if path is a G path
    elif cphi2 == 1:
        
        print('Path is of type G.')
        
        phi2 = 0
        phi3 = 0
        phi1 = math.atan2(alpha21, alpha11) # Since path is a G path
        
        # Testing if the final configuration obtained from the C path is the
        # same as the desired final configuration
        _, _, _, fin_config_path, _, _, _ =\
            points_path(np.identity(3), r_mod, r_mod, 1, [phi1, phi2, phi3], path_type)
            
        # Checking if the minimum and maximum value in the difference in the final
        # configurations is small
        if abs(max(map(max, fin_config_path - fin_config_mod))) <= 10**(-6)\
            and abs(min(map(min, fin_config_path - fin_config_mod))) <= 10**(-6):
                
            path_params.append([R*phi1, phi1, phi2, phi3, R*phi1])
            
        else:
            
            print(path_type.upper() + ' path does not exist.')
            path_params.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
            
    else:
        
        # Obtaining the two solutions for phi2
        # Checking if cphi2 = -1.
        if cphi2 == -1:
            
            phi2_array = [math.pi]
        
        else:
        
            phi2_array = [math.acos(cphi2), 2*math.pi - math.acos(cphi2)]
        
        
        # Running through each solution and obtaining the corresponding solution
        # for phi1 and phi3
        for phi2 in phi2_array:
            
            # Obtaining the solutions for phi1 and phi3
            den = (sin(phi2))**2 + r_mod**2*(1 - cphi2)**2
            sphi1 = (alpha13*sin(phi2) + alpha23*r_mod*(1 - cphi2))/(math.sqrt(1 - r_mod**2)*den)
            cphi1 = (-alpha23*sin(phi2) + alpha13*r_mod*(1 - cphi2))/(math.sqrt(1 - r_mod**2)*den)
            sphi3 = (alpha31*sin(phi2) - alpha32*r_mod*(1 - cphi2))/(math.sqrt(1 - r_mod**2)*den)
            cphi3 = (alpha32*sin(phi2) + alpha31*r_mod*(1 - cphi2))/(math.sqrt(1 - r_mod**2)*den)
            
            # If path type is grg, we have the same expressions, but multiplied by -1.
            if path_type == 'grg':
                
                sphi1 = sphi1*(-1)
                cphi1 = cphi1*(-1)
                sphi3 = sphi3*(-1)
                cphi3 = cphi3*(-1)
            
            # Obtaining the angles
            phi1 = np.mod(math.atan2(sphi1, cphi1), 2*math.pi)
            phi3 = np.mod(math.atan2(sphi3, cphi3), 2*math.pi)
            
            _, _, _, fin_config_path, _, _, _ = \
                points_path(np.identity(3), r_mod, r_mod, 1, [phi1, phi2, phi3], path_type)
                            
            # Checking if the minimum and maximum value in the difference in the final
            # configurations is small
            if abs(max(map(max, fin_config_path - fin_config_mod))) <= 10**(-6)\
                and abs(min(map(min, fin_config_path - fin_config_mod))) <= 10**(-6):   
                
                U = math.sqrt(((R/r)**2 - 1))
                
                # Appending the path length, path parameters, and path cost
                path_cost = R*(phi1 + phi3) + r*(1 + mu*U)*phi2
                path_length = R*(phi1 + phi3) + r*phi2
                    
                path_params.append([path_length, phi1, phi2, phi3, path_cost])
                            
        # Checking if no solution was obtained for the path
        if len(path_params) == 0:
            
            print(path_type.upper() + ' path does not exist.')
            path_params.append([np.NaN, np.NaN, np.NaN, np.NaN])
                
    return path_params

def path_generation_sphere_CCG(ini_config, fin_config, rL, rR, R, muL, muR, path_type = 'lrg'):
    
    # Modifying the configurations and the parameters of the turn
    fin_config_mod = modifying_initial_final_configurations_unit_sphere(ini_config, fin_config, R)
    rL_mod = rL/R
    rR_mod = rR/R
    
    Ul = math.sqrt(((R/rL)**2 - 1))
    Ur = math.sqrt(((R/rR)**2 - 1))
    
    path_types = ['lrg', 'rlg']
    
    if path_type not in path_types:
        
        raise Exception('Incorrect path type passed.')
        
    # Defining variables corresponding to the final configuration
    alpha11 = fin_config_mod[0, 0]; alpha12 = fin_config_mod[0, 1]; alpha13 = fin_config_mod[0, 2];
    alpha21 = fin_config_mod[1, 0]; alpha22 = fin_config_mod[1, 1]; alpha23 = fin_config_mod[1, 2];
    alpha31 = fin_config_mod[2, 0]; alpha32 = fin_config_mod[2, 1]; alpha33 = fin_config_mod[2, 2];
    
    # Storing the details of the path
    path_params = []
    
    if path_type in path_types:
        
        if path_type == 'lrg':
            
            cphi2 = (sqrt(1 - rL_mod**2)*alpha13 + rL_mod*alpha33\
                     + rR_mod*(sqrt(1 - rL_mod**2)*sqrt(1 - rR_mod**2) - rL_mod*rR_mod))/\
                (sqrt(1 - rR_mod**2)*(rR_mod*sqrt(1 - rL_mod**2) + rL_mod*sqrt(1 - rR_mod**2)))
                
        # Obtaining the two solutions (if they exist) for phi2. Accounting for
        # numerical inaccuracies
        if abs(cphi2) > 1 and abs(cphi2) <= 1 + 10**(-6):
        
            cphi2 = np.sign(cphi2)
            
        # Checking if path exists
        if abs(cphi2) > 1:
            
            print(path_type.upper() + ' path does not exist as cphi2 is ' + str(cphi2) + '.')
            path_length = np.NaN; phi1 = np.NaN; phi2 = np.NaN; phi3 = np.NaN;
            path_params.append([path_length, phi1, phi2, phi3, np.NaN])
            
            return path_params
        
        else:
            
            # One or two possible solutions exist for phi2
            if abs(cphi2) == 1:
                
                phi2_array = [math.acos(cphi2)]
                
            else:
            
                phi2_array = [math.acos(cphi2), 2*math.pi - math.acos(cphi2)]
            
            print('Solutions for phi2 for path ' + str(path_type) + ' are ' + str(phi2_array))
            
            # Obtaining the possible solutions for phi1 and phi3 corresponding to
            # each phi2
            for phi2 in phi2_array:
                
                if path_type == 'lrg':
                    
                    # Obtaining the solutions for phi1
                    
                    A = sqrt(1 - rR_mod**2)*(sqrt(1 - rL_mod**2)*sqrt(1 - rR_mod**2) - rL_mod*rR_mod)*cos(phi2)\
                        + rR_mod*(rL_mod*sqrt(1 - rR_mod**2) + rR_mod*sqrt(1 - rL_mod**2))
                    B = sin(phi2)*sqrt(1 - rR_mod**2)
                    C = (alpha33 - rL_mod*sqrt(1 - rR_mod**2)*cos(phi2)*(rR_mod*sqrt(1 - rL_mod**2)\
                                                                         + rL_mod*sqrt(1 - rR_mod**2))\
                         + rL_mod*rR_mod*(sqrt(1 - rL_mod**2)*sqrt(1 - rR_mod**2) - rL_mod*rR_mod))/sqrt(1 - rL_mod**2)
                    
                    if sqrt(A**2 + B**2) <= 10**(-6):
                        
                        continue # Since phi1 cannot be solved
                        
                    else:
                        
                        beta = math.atan2(B, A) # Computing the angle to be used to solve for phi1
                        
                        # Obtaining the RHS for solving for phi1
                        phi1_RHS = C/sqrt(A**2 + B**2)
                        
                        # Checking if phi1_RHS is in [-1, 1] with a tolerance
                        if abs(phi1_RHS) > 1 and abs(phi1_RHS) <= 1 + 10**(-6):
                            
                            phi1_RHS = np.sign(phi1_RHS)
                            
                        if abs(phi1_RHS) > 1:
                            
                            continue # Since solution for phi1 does not exist
                        
                        # Obtaining the solutions for phi1
                        phi1_soln_arr = [np.mod(math.acos(phi1_RHS) + beta, 2*math.pi),\
                                         np.mod(2*math.pi - math.acos(phi1_RHS) + beta, 2*math.pi)]
                            
                        # Checking if either solution is nearly equal to zero or 2*math.pi
                        if 2*math.pi - phi1_soln_arr[0] <= 10**(-6) or phi1_soln_arr[0] <= 10**(-6):
                            
                            phi1_soln_arr[0] = 0
                            
                        if 2*math.pi - phi1_soln_arr[1] <= 10**(-6) or phi1_soln_arr[1] <= 10**(-6):
                            
                            phi1_soln_arr[1] = 0
                            
                        # Checking if the two solutions are equal
                        if abs(phi1_soln_arr[1] - phi1_soln_arr[0]) <= 10**(-6):
                            
                            phi1_soln_arr = [np.mod(math.acos(phi1_RHS) + beta, 2*math.pi)]
                            
                        print('Solutions for phi1 are ' + str(phi1_soln_arr))
                        
                    # Obtaining the solutions for phi3
                    D = sqrt(1 - rL_mod**2) - rR_mod*(1 - cos(phi2))*(rR_mod*sqrt(1 - rL_mod**2)\
                                                                      + rL_mod*sqrt(1 - rR_mod**2))
                    E = (rR_mod*sqrt(1 - rL_mod**2) + rL_mod*sqrt(1 - rR_mod**2))*sin(phi2)
                    F = (alpha11 + rL_mod*sqrt(1 - rL_mod**2)*(alpha13 + alpha31) + rL_mod**2*(alpha33 - alpha11)\
                         - rL_mod**2*cos(phi2)\
                         - rL_mod*rR_mod*(1 - cos(phi2))*(rL_mod*rR_mod - sqrt(1 - rL_mod**2)*sqrt(1 - rR_mod**2)))/\
                        sqrt(1 - rL_mod**2)
                    
                    if sqrt(D**2 + E**2) <= 10**(-6):
                        
                        continue # Since phi3 cannot be solved for
                        
                    else:
                        
                        gamma = math.atan2(E, D)
                        
                        # Checking if the RHS for solving for phi3 is in [-1, 1] with a tolerance
                        phi3_RHS = F/sqrt(D**2 + E**2)
                        
                        if abs(phi3_RHS) > 1 and abs(phi3_RHS) <= 1 + 10**(-6):
                            
                            phi3_RHS = np.sign(phi3_RHS)
                            
                        if abs(phi3_RHS) > 1:
                            
                            continue # Since solution for phi3 does not exist
                            
                        # Obtaining the solutions for phi3
                        phi3_soln_arr = [np.mod(math.acos(phi3_RHS) - gamma, 2*math.pi),\
                                         np.mod(2*math.pi - math.acos(phi3_RHS) - gamma, 2*math.pi)]
                            
                        # Checking if either solution is nearly equal to zero or 2*math.pi
                        if 2*math.pi - phi3_soln_arr[0] <= 10**(-6) or phi3_soln_arr[0] <= 10**(-6):
                            
                            phi3_soln_arr[0] = 0
                            
                        if 2*math.pi - phi3_soln_arr[1] <= 10**(-6) or phi3_soln_arr[1] <= 10**(-6):
                            
                            phi3_soln_arr[1] = 0
                            
                        # Checking if the two solutions are equal
                        if abs(phi3_soln_arr[1] - phi3_soln_arr[0]) <= 10**(-6):
                            
                            phi3_soln_arr = [np.mod(math.acos(phi3_RHS) - gamma, 2*math.pi)]
                            
                        print('Solutions for phi3 are ' + str(phi3_soln_arr))
                        
                    # Running through the solutions for phi1 and phi3, and checking if the final
                    # configuration is reached
                    for phi1 in phi1_soln_arr:
                        
                        for phi3 in phi3_soln_arr:
                            
                            # Obtaining the final configuration of the path
                            _, _, _, fin_config_path, _, _, _ =\
                                points_path(np.identity(3), rL_mod, rR_mod, 1, [phi1, phi2, phi3], path_type)
                                
                            # Checking if the minimum and maximum value in the difference in the final
                            # configurations is small
                            if abs(max(map(max, fin_config_path - fin_config_mod))) <= 10**(-6)\
                                and abs(min(map(min, fin_config_path - fin_config_mod))) <= 10**(-6):
                                
                                if path_type == 'lrg':    
                                
                                    # Appending the path length, path parameters, and path cost
                                    path_cost = R*phi3 + rL*(1 + muL*Ul)*phi1 + rR*(1 + muR*Ur)*phi2
                                    path_length = R*phi3 + rL*phi1 + rR*phi3
                                    
                                path_params.append([path_length, phi1, phi2, phi3, path_cost])
                                
        # Checking if no solution was obtained for the path
        if len(path_params) == 0:
            
            print(path_type.upper() + ' path does not exist.')
            path_params.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
    
    return path_params

# def path_generation_sphere_CG_GC_special_cases_mu(ini_config, fin_config, mu, path_type = 'lgr'):
    
#     # Modifying the configurations and the parameters of the turn
#     fin_config_mod = modifying_initial_final_configurations_unit_sphere(ini_config, fin_config, 1)
    
#     Ul = mu
#     Ur = mu
    
#     path_types = ['lg', 'rg', 'gl', 'gr']
    
#     if path_type not in path_types:
        
#         raise Exception('Incorrect path type passed.')
        
#     # Defining variables corresponding to the final configuration
#     alpha11 = fin_config_mod[0, 0]; alpha12 = fin_config_mod[0, 1]; alpha13 = fin_config_mod[0, 2];
#     alpha21 = fin_config_mod[1, 0]; alpha22 = fin_config_mod[1, 1]; alpha23 = fin_config_mod[1, 2];
#     alpha31 = fin_config_mod[2, 0]; alpha32 = fin_config_mod[2, 1]; alpha33 = fin_config_mod[2, 2];
    
#     # Storing the details of the path
#     path_params = []
    
#     # if path_type in path_types:
        
#     #     if path_type == 'lg':
            
            

def path_generation_sphere_three_seg(ini_config, fin_config, rL, rR, R, muL, muR, path_type = 'rgr'):
    
    # Modifying the configurations and the parameters of the turn
    fin_config_mod = modifying_initial_final_configurations_unit_sphere(ini_config, fin_config, R)
    rL_mod = rL/R
    rR_mod = rR/R
    
    Ul = math.sqrt(((R/rL)**2 - 1))
    Ur = math.sqrt(((R/rR)**2 - 1))
    
    path_types = ['lgl', 'rgr', 'lgr', 'rgl', 'lrl', 'rlr']
    
    if path_type not in path_types:
        
        raise Exception('Incorrect path type passed.')
        
    # Defining variables corresponding to the final configuration
    alpha11 = fin_config_mod[0, 0]; alpha12 = fin_config_mod[0, 1]; alpha13 = fin_config_mod[0, 2];
    alpha21 = fin_config_mod[1, 0]; alpha22 = fin_config_mod[1, 1]; alpha23 = fin_config_mod[1, 2];
    alpha31 = fin_config_mod[2, 0]; alpha32 = fin_config_mod[2, 1]; alpha33 = fin_config_mod[2, 2];
    
    # Storing the details of the path
    path_params = []
    
    if path_type in path_types:
        
        if path_type == 'lgl':
            
            cphi2 = (alpha11 + rL_mod*math.sqrt(1 - rL_mod**2)*(alpha13 + alpha31)\
                     + rL_mod**2*(alpha33 - alpha11 - 1))/(1 - rL_mod**2)
        
        elif path_type == 'lgr':
            
            cphi2 = (math.sqrt((1 - rL_mod**2)*(1 - rR_mod**2))*alpha11\
                     + rL_mod*math.sqrt(1 - rR_mod**2)*alpha31\
                     - rR_mod*math.sqrt(1 - rL_mod**2)*alpha13\
                     + rL_mod*rR_mod*(1 - alpha33))/(math.sqrt((1 - rL_mod**2)*(1 - rR_mod**2)))
        
        elif path_type == 'rgl':
            
            cphi2 = (math.sqrt((1 - rL_mod**2)*(1 - rR_mod**2))*alpha11\
                     + rL_mod*math.sqrt(1 - rR_mod**2)*alpha13\
                     - rR_mod*math.sqrt(1 - rL_mod**2)*alpha31\
                     + rL_mod*rR_mod*(1 - alpha33))/(math.sqrt((1 - rL_mod**2)*(1 - rR_mod**2)))
            # print(cphi2)
        
        elif path_type == 'rgr':
        
            cphi2 = (alpha11 - rR_mod*math.sqrt(1 - rR_mod**2)*(alpha13 + alpha31)\
                     + rR_mod**2*(alpha33 - alpha11 - 1))/(1 - rR_mod**2)
                
        elif path_type == 'lrl':
            
            cphi2 = (alpha11 + rL_mod*sqrt(1 - rL_mod**2)*(alpha13 + alpha31)\
                     + rL_mod**2*(alpha33 - alpha11) - 1)\
                /(rL_mod**2 + rR_mod**2 - 2*rL_mod*rR_mod*(rL_mod*rR_mod - sqrt(1 - rL_mod**2)*sqrt(1 - rR_mod**2)))\
                    + 1
                    
        elif path_type == 'rlr':
            
            cphi2 = (alpha11 - rR_mod*sqrt(1 - rR_mod**2)*(alpha13 + alpha31)\
                     + rR_mod**2*(alpha33 - alpha11) - 1)\
                /(rL_mod**2 + rR_mod**2 - 2*rL_mod*rR_mod*(rL_mod*rR_mod - sqrt(1 - rL_mod**2)*sqrt(1 - rR_mod**2)))\
                    + 1
        
        # Obtaining the two solutions (if they exist) for phi2. Accounting for
        # numerical inaccuracies
        if abs(cphi2) > 1 and abs(cphi2) <= 1 + 10**(-8):
        
            cphi2 = np.sign(cphi2)
            
        elif abs(cphi2) >= 1 - 10**(-8):
            
            cphi2 = np.sign(cphi2)
            
        # Checking if path exists
        if abs(cphi2) > 1:
            
            # print(path_type.upper() + ' path does not exist as cphi2 is ' + str(cphi2) + '.')
            path_length = np.NaN; phi1 = np.NaN; phi2 = np.NaN; phi3 = np.NaN;
            path_params.append([path_length, phi1, phi2, phi3, np.NaN])
            
            return path_params
        
        # Checking if the path is a degenerate 'C' if lgl or rgr path
        elif cphi2 == 1 and (path_type in ['lgl', 'rgr', 'lrl', 'rlr']):
            
            # print('Path is of type ' + path_type[0].upper())
            
            # Setting the second and third angles to zero. Third angle is set to zero
            # since the first and third arcs of the same type.
            phi2 = 0
            phi3 = 0
            
            if path_type == 'lgl' or path_type == 'lrl':
                
                phi1 = math.atan2(alpha21, rL_mod*alpha22)
            
            else:
            
                phi1 = math.atan2(alpha21, rR_mod*alpha22)
                
            # Testing if the final configuration obtained from the C path is the
            # same as the desired final configuration
            _, _, _, fin_config_path, _, _, _ =\
                points_path(np.identity(3), rL_mod, rR_mod, 1, [phi1, phi2, phi3], path_type)
                
            # Checking if the minimum and maximum value in the difference in the final
            # configurations is small
            if abs(max(map(max, fin_config_path - fin_config_mod))) <= 10**(-8)\
                and abs(min(map(min, fin_config_path - fin_config_mod))) <= 10**(-8):
                
                if path_type == 'lgl' or path_type == 'lrl':    
                
                    path_params.append([rL*phi1, phi1, phi2, phi3, (rL + muL*Ul*rL)*phi1])
                    
                elif path_type == 'rgr' or path_type == 'rlr':
                    
                    path_params.append([rR*phi1, phi1, phi2, phi3, (rR + muR*Ur*rR)*phi1])                    
               
            else:
                
                # print(path_type.upper() + ' path does not exist.')
                path_params.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])   
                               
        else:
            
            # One or two possible solutions exist for phi2
            if abs(cphi2) == 1:
                
                phi2_array = [math.acos(cphi2)]
                
            else:
            
                phi2_array = [math.acos(cphi2), 2*math.pi - math.acos(cphi2)]
            
            # print('Solutions for phi2 for path ' + str(path_type) + ' are ' + str(phi2_array))
            
            # Obtaining the possible solutions for phi1 and phi3 corresponding to
            # each phi2
            for phi2 in phi2_array:
                
                # print('phi2 considered is ' + str(phi2))
                # Obtaining possible solutions for phi1 and phi3
                if path_type == 'lgl':
                    
                    phi1RHS = ((alpha33 - alpha11)*rL_mod - alpha13*rL_mod**2*(1 - rL_mod**2)**(-0.5)\
                               + alpha31*math.sqrt(1 - rL_mod**2))/math.sqrt(rL_mod**2*(1 - cos(phi2))**2 + (sin(phi2))**2)
                        
                    beta = math.atan2(sin(phi2), rL_mod*(1 - cos(phi2)))
                    
                    phi3RHS = ((alpha33 - alpha11)*rL_mod + alpha13*math.sqrt(1 - rL_mod**2)\
                               - alpha31*rL_mod**2*(1 - rL_mod**2)**(-0.5))/math.sqrt(rL_mod**2*(1 - cos(phi2))**2 + (sin(phi2))**2)
                        
                    gamma = beta
                
                elif path_type == 'lgr':
                    
                    if rL_mod == rR_mod and abs(phi2 - math.pi) <= 10**(-6):
                        
                        if muL <= muR:
                            
                            phi1 = np.mod(math.atan2(alpha12, -rL_mod*alpha22), 2*math.pi)
                            phi3 = 0
                            phi2 = math.pi
                            
                        else:
                            
                            phi1 = 0
                            phi3 = np.mod(math.atan2(alpha12, -rL_mod*alpha22), 2*math.pi)
                            phi2 = math.pi
                        
                    else:
                        
                        phi1RHS = (rL_mod*math.sqrt(1 - rR_mod**2)*alpha11 - math.sqrt((1 - rL_mod**2)*(1 - rR_mod**2))*alpha31\
                                   - rL_mod*rR_mod*alpha13 + rR_mod*math.sqrt(1 - rL_mod**2)*alpha33)/\
                                  (math.sqrt((rL_mod*math.sqrt(1 - rR_mod**2)*cos(phi2) + rR_mod*math.sqrt(1 - rL_mod**2))**2\
                                             + (1 - rR_mod**2)*(sin(phi2))**2))
                        
                        # Including a negative sign in addition for beta and gamma, since
                        # beta and gamma are added to the angle corr. to RHS further down,
                        # but analytically, beta and gamma should be subtracted
                        beta = -math.atan2(math.sqrt(1 - rR_mod**2)*sin(phi2),\
                                           (rL_mod*math.sqrt(1 - rR_mod**2)*cos(phi2) + rR_mod*math.sqrt(1 - rL_mod**2)))
                        
                        phi3RHS = (rR_mod*math.sqrt(1 - rL_mod**2)*alpha11 + math.sqrt((1 - rL_mod**2)*(1 - rR_mod**2))*alpha13\
                                   + rL_mod*rR_mod*alpha31 + rL_mod*math.sqrt(1 - rR_mod**2)*alpha33)/\
                                  (math.sqrt((rR_mod*math.sqrt(1 - rL_mod**2)*cos(phi2) + rL_mod*math.sqrt(1 - rR_mod**2))**2\
                                             + (1 - rL_mod**2)*(sin(phi2))**2))
                        
                        gamma = -math.atan2(math.sqrt(1 - rL_mod**2)*sin(phi2),\
                                           (rR_mod*math.sqrt(1 - rL_mod**2)*cos(phi2) + rL_mod*math.sqrt(1 - rR_mod**2)))
                            
                elif path_type == 'rgl':
                    
                    if rL_mod == rR_mod and abs(phi2 - math.pi) <= 10**(-6):
                        
                        if muL <= muR:
                            
                            phi3 = np.mod(math.atan2(alpha12, -rL_mod*alpha22), 2*math.pi)
                            phi1 = 0
                            phi2 = math.pi
                            
                        else:
                            
                            phi3 = 0
                            phi1 = np.mod(math.atan2(alpha12, -rL_mod*alpha22), 2*math.pi)
                            phi2 = math.pi
                        
                    else:
                        
                        phi1RHS = (rR_mod*math.sqrt(1 - rL_mod**2)*alpha11 + math.sqrt((1 - rL_mod**2)*(1 - rR_mod**2))*alpha31\
                                   + rL_mod*rR_mod*alpha13 + rL_mod*math.sqrt(1 - rR_mod**2)*alpha33)/\
                                  (math.sqrt((rR_mod*math.sqrt(1 - rL_mod**2)*cos(phi2) + rL_mod*math.sqrt(1 - rR_mod**2))**2\
                                             + (1 - rL_mod**2)*(sin(phi2))**2))
                        
                        # Including a negative sign in addition for beta and gamma, since
                        # beta and gamma are added to the angle corr. to RHS further down,
                        # but analytically, beta and gamma should be subtracted
                        beta = -math.atan2(math.sqrt(1 - rL_mod**2)*sin(phi2),\
                                           (rR_mod*math.sqrt(1 - rL_mod**2)*cos(phi2) + rL_mod*math.sqrt(1 - rR_mod**2)))
                                      
                        phi3RHS = (rL_mod*math.sqrt(1 - rR_mod**2)*alpha11 - math.sqrt((1 - rL_mod**2)*(1 - rR_mod**2))*alpha13\
                                   - rL_mod*rR_mod*alpha31 + rR_mod*math.sqrt(1 - rL_mod**2)*alpha33)/\
                                  (math.sqrt((rL_mod*math.sqrt(1 - rR_mod**2)*cos(phi2) + rR_mod*math.sqrt(1 - rL_mod**2))**2\
                                             + (1 - rR_mod**2)*(sin(phi2))**2))
                        
                        gamma = -math.atan2(math.sqrt(1 - rR_mod**2)*sin(phi2),\
                                           (rL_mod*math.sqrt(1 - rR_mod**2)*cos(phi2) + rR_mod*math.sqrt(1 - rL_mod**2)))
                      
                
                elif path_type == 'rgr':
                
                    phi1RHS = ((alpha33 - alpha11)*rR_mod + alpha13*rR_mod**2*(1 - rR_mod**2)**(-0.5)\
                               - alpha31*math.sqrt(1 - rR_mod**2))/math.sqrt(rR_mod**2*(1 - cos(phi2))**2 + (sin(phi2))**2)
                        
                    beta = math.atan2(sin(phi2), rR_mod*(1 - cos(phi2)))
                    
                    phi3RHS = ((alpha33 - alpha11)*rR_mod - alpha13*math.sqrt(1 - rR_mod**2)\
                               + alpha31*rR_mod**2*(1 - rR_mod**2)**(-0.5))/math.sqrt(rR_mod**2*(1 - cos(phi2))**2 + (sin(phi2))**2)
                        
                    gamma = beta
                    
                elif path_type == 'lrl':
                    
                    A = -1 + 2*rL_mod**2 + rL_mod**2*(1 - 2*rL_mod**2)*(1 - cos(phi2))\
                        + rR_mod**2*(1 - cos(phi2))\
                        + 2*rL_mod*rR_mod*(1 - 2*rL_mod**2)*sqrt(1 - rL_mod**2)*sqrt(1 - rR_mod**2)*(1 - cos(phi2))\
                        - 4*rL_mod**2*rR_mod**2*(1 - cos(phi2))*(1 - rL_mod**2)
                    B = (rL_mod*(2*rR_mod**2 - 1)*sqrt(1 - rL_mod**2)\
                         + rR_mod*(2*rL_mod**2 - 1)*sqrt(1 - rR_mod**2))*(1 - cos(phi2))
                    C = (rR*sqrt(1 - rL_mod**2) + rL_mod*sqrt(1 - rR_mod**2))*sin(phi2)
                    
                    phi1RHS = (-alpha11 + rL_mod**2*(alpha11 + alpha33) + rL_mod*sqrt(1 - rL_mod**2)*(alpha31 - alpha13) - A)/\
                        (2*rL_mod*sqrt(1 - rL_mod**2)*sqrt(B**2 + C**2))
                        
                    beta = math.atan2(C, B)
                    
                    phi3RHS = (-alpha11 + rL_mod**2*(alpha11 + alpha33) - rL_mod*sqrt(1 - rL_mod**2)*(alpha31 - alpha13) - A)/\
                        (2*rL_mod*sqrt(1 - rL_mod**2)*sqrt(B**2 + C**2))
                        
                    gamma = beta

                elif path_type == 'rlr':

                    A = (2*rR_mod**2 - 1)*(1 - rR_mod**2) + (1 - 2*rR_mod**2)*sqrt(1 - rR_mod**2)*rL_mod*(1 - cos(phi2))\
                        *(rL_mod*sqrt(1 - rR_mod**2) + rR_mod*sqrt(1 - rL_mod**2))\
                        + rR_mod*(2*rR_mod**2 - 1)*(rR_mod*cos(phi2) + rL_mod*(1 - cos(phi2))*(rL_mod*rR_mod - sqrt(1 - rL_mod**2)*sqrt(1 - rR_mod**2)))
                    B = rR_mod**2*sqrt(1 - rR_mod**2)*(-sqrt(1 - rR_mod**2) + (1 - cos(phi2))*rL_mod*(rL_mod*sqrt(1 - rR_mod**2) + rR_mod*sqrt(1 - rL_mod**2)))\
                        + rR_mod*(1 - rR_mod**2)*(rR_mod*cos(phi2) + rL_mod*(1 - cos(phi2))*(rL_mod*rR_mod - sqrt(1 - rL_mod**2)*sqrt(1 - rR_mod**2)))
                    C = sin(phi2)*rR_mod*sqrt(1 - rR_mod**2)*(rL_mod*sqrt(1 - rR_mod**2) + rR_mod*sqrt(1 - rL_mod**2))

                    phi1RHS = (-alpha11 + rR_mod**2*(alpha11 + alpha33) - rR_mod*sqrt(1 - rR_mod**2)*(alpha31 - alpha13) - A)\
                        /(2*sqrt(B**2 + C**2))
                    
                    beta = math.atan2(C, B)

                    phi3RHS = (-alpha11 + rR_mod**2*(alpha11 + alpha33) + rR_mod*sqrt(1 - rR_mod**2)*(alpha31 - alpha13) - A)\
                        /(2*sqrt(B**2 + C**2))
                    
                    gamma = beta
                    
                if path_type in ['lgr', 'rgl'] and rL_mod == rR_mod and phi2 == math.pi:
                    
                    phi1_array = [phi1]
                    phi3_array = [phi3]
                
                else:
                
                    # Checking if solution for phi1 and phi3 can be obtained withih
                    # a tolerance
                    if abs(phi1RHS) > 1 and abs(phi1RHS) <= 1 + 10**(-8):
            
                        phi1RHS = np.sign(phi1RHS)
                        
                    if abs(phi3RHS) > 1 and abs(phi3RHS) <= 1 + 10**(-8):
            
                        phi3RHS = np.sign(phi3RHS)
                        
                    # Checking condition for phi1 and phi3 cannot be solved for
                    if abs(phi1RHS) > 1 or abs(phi3RHS) > 1:
                        
                        continue # Skipping the possible solution for phi2 as it is not a viable solution
                        
                    # Checking if one or two solutions exist for phi1
                    if abs(phi1RHS) == 1:
                        
                        # Only one solution exists for phi1
                        phi1_array = [np.mod(math.acos(phi1RHS) + beta, 2*math.pi)]
                        
                        # Checking if the angle is nearly 2*math.pi or zero
                        if 2*math.pi - phi1_array[0] <= 10**(-8) or phi1_array[0] <= 10**(-8):
                            
                            phi1_array[0] = 0
                        
                    else:
                        
                        # Obtaining the two possible solutions for phi1
                        phi1_array = [np.mod(math.acos(phi1RHS) + beta, 2*math.pi),\
                                      np.mod(2*math.pi - math.acos(phi1RHS) + beta, 2*math.pi)]
                        
                        # Checking if one of the solutions is nearly 2*math.pi or zero
                        if 2*math.pi - phi1_array[0] <= 10**(-8) or phi1_array[0] <= 10**(-8):
                            
                            phi1_array[0] = 0
                            
                        if 2*math.pi - phi1_array[1] <= 10**(-8) or phi1_array[1] <= 10**(-8):
                            
                            phi1_array[1] = 0
                            
                        # Checking if the two solutions are the same
                        if abs(phi1_array[0] - phi1_array[1]) <= 10**(-8):
                            
                            phi1_array = [np.mod(math.acos(phi1RHS) + beta, 2*math.pi)]
                    
                    # Checking if one or two solutions exist for phi3
                    if abs(phi3RHS) == 1:
                        
                        # Only one solution exists for phi1
                        phi3_array = [np.mod(math.acos(phi3RHS) + gamma, 2*math.pi)]
                        
                        # Checking if one of the angles is nearly 2*math.pi or zero
                        if 2*math.pi - phi3_array[0] <= 10**(-8) or phi3_array[0] <= 10**(-8):
                            
                            phi3_array[0] = 0
                        
                    else:
                        
                        # Obtaining the two possible solutions for phi3
                        phi3_array = [np.mod(math.acos(phi3RHS) + gamma, 2*math.pi),\
                                      np.mod(2*math.pi - math.acos(phi3RHS) + gamma, 2*math.pi)]
                            
                        # Checking if one of the solutions is nearly 2*math.pi or zero
                        if 2*math.pi - phi3_array[0] <= 10**(-8) or phi3_array[0] <= 10**(-8):
                            
                            phi3_array[0] = 0
                            
                        if 2*math.pi - phi3_array[1] <= 10**(-8) or phi3_array[1] <= 10**(-8):
                            
                            phi3_array[1] = 0
                            
                        # Checking if the two solutions are the same
                        if abs(phi3_array[0] - phi3_array[1]) <= 10**(-8):
                            
                            phi3_array = [np.mod(math.acos(phi3RHS) + gamma, 2*math.pi)]
                
                # print('Solutions for phi 1 are ' + str(phi1_array))
                # print('Solutions for phi 3 are ' + str(phi3_array))
                
                # From all possible solutions for phi1 and phi3 for the chosen phi2,
                # identifying those solutions that connect to the chosen final configuration
                for phi1 in phi1_array:
                    
                    for phi3 in phi3_array:
                        
                        # Obtaining the final configuration of the path
                        _, _, _, fin_config_path, _, _, _ =\
                            points_path(np.identity(3), rL_mod, rR_mod, 1, [phi1, phi2, phi3], path_type)
                        
                        # print(fin_config_path)
                            
                        # Checking if the minimum and maximum value in the difference in the final
                        # configurations is small
                        if abs(max(map(max, fin_config_path - fin_config_mod))) <= 10**(-8)\
                            and abs(min(map(min, fin_config_path - fin_config_mod))) <= 10**(-8):
                            
                            if path_type == 'lgl':    
                            
                                # Appending the path length, path parameters, and path cost
                                path_cost = rL*(1 + muL*Ul)*(phi1 + phi3) + R*phi2
                                path_length = rL*(phi1 + phi3) + R*phi2
                                
                            elif path_type == 'lgr':
                                
                                # Appending the path length, path parameters, and path cost
                                path_cost = rL*(1 + muL*Ul)*phi1 + rR*(1 + muR*Ur)*phi3 + R*phi2
                                path_length = rL*phi1 + rR*phi3 + R*phi2
                                
                            elif path_type == 'rgl':
                                
                                # Appending the path length, path parameters, and path cost
                                path_cost = rL*(1 + muL*Ul)*phi3 + rR*(1 + muR*Ur)*phi1 + R*phi2
                                path_length = rL*phi3 + rR*phi1 + R*phi2
                                
                            elif path_type == 'rgr':
                                
                                # Appending the path length, path parameters, and path cost
                                path_cost = rR*(1 + muR*Ur)*(phi1 + phi3) + R*phi2
                                path_length = rR*(phi1 + phi3) + R*phi2
                                
                            elif path_type == 'lrl':
                                
                                # Appending the path length, path parameters, and path cost
                                path_cost = rL*(1 + muL*Ul)*(phi1 + phi3) + rR*(1 + muR*Ur)*phi2
                                path_length = rL*(phi1 + phi3) + rR*phi2

                            elif path_type == 'rlr':

                                # Appending the path length, path parameters, and path cost
                                path_cost = rR*(1 + muR*Ur)*(phi1 + phi3) + rL*(1 + muL*Ul)*phi2
                                path_length = rR*(phi1 + phi3) + rL*phi2
                                
                            path_params.append([path_length, phi1, phi2, phi3, path_cost])
                            
            # print('Parameters of the ' + str(path_type) + ' path.')
            # print(path_params)
                            # print('Solution for phi2 number ' + str(phi2_array.index(phi2)) +\
                            #       ', solution for phi1 number ' + str(phi1_array.index(phi1)) +\
                            #       ', solution for phi3 number ' + str(phi3_array.index(phi3)))
                            
            # Checking if no solution was obtained for the path
            if len(path_params) == 0:
                
                print(path_type.upper() + ' path does not exist.')
                path_params.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
                
    return path_params

def path_generation_sphere_four_seg(ini_config, fin_config, rL, rR, R, muL, muR, path_type = 'glgr'):
    
    # Modifying the configurations and the parameters of the turn
    fin_config_mod = modifying_initial_final_configurations_unit_sphere(ini_config, fin_config, R)
    rL_mod = rL/R
    rR_mod = rR/R
    
    Ul = math.sqrt(((R/rL)**2 - 1))
    Ur = math.sqrt(((R/rR)**2 - 1))
    
    path_types = ['lgrg', 'grgl', 'rglg', 'glgr']
    
    if path_type not in path_types:
        
        raise Exception('Incorrect path type passed.')
        
    # Path construction for muL = muR, and muL <= Ul, muR <= Ur
    if muL != muR or muL > Ul or muR > Ur:
        
        raise Exception('Conditions for path generation are not met.')
        
    # Defining variables corresponding to the final configuration
    alpha11 = fin_config_mod[0, 0]; alpha12 = fin_config_mod[0, 1]; alpha13 = fin_config_mod[0, 2];
    alpha21 = fin_config_mod[1, 0]; alpha22 = fin_config_mod[1, 1]; alpha23 = fin_config_mod[1, 2];
    alpha31 = fin_config_mod[2, 0]; alpha32 = fin_config_mod[2, 1]; alpha33 = fin_config_mod[2, 2];
    
    # Storing the details of the path
    path_params = []
    
    mu = muL
    
    if path_type == 'glgr':
        
        # Obtaining the solutions for phi3 (if they exist).
        A = -2*mu*Ul*math.sqrt(1 - rR_mod**2)*math.sqrt(1 + Ul**2)
        B = mu**2*(1 + Ul**2)*(rL_mod*math.sqrt(1 - rR_mod**2) + rR_mod*math.sqrt(1 - rL**2)\
                               - (math.sqrt(1 - rR_mod**2)*alpha31 - rR_mod*(alpha33 - 1))/(2*math.sqrt(1 - rL_mod**2)))
        C = ((math.sqrt(1 - rR_mod**2)*alpha31 - rR_mod*(alpha33 - 1))/(2*math.sqrt(1 - rL_mod**2)))*(Ul - mu)**2
        
        # Checking if the discriminant for the quadratic equation is zero, less than
        # zero, or greater than zero
        D = (A + B + C)**2 - 4*A*C
        
        # Using a tolerance to check if D = 0, D < 0, or D > 0.
        if np.sign(D) == -1 and abs(D) <= 10**(-6):
            
            D = 0
        
        if D < 0: # Path does not exist if the discriminant is negative
        
            return [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
        
        elif D > 0: # Checking if the discriminant is positive
        
            x = [((A + B + C + math.sqrt(D))/(2*A)), (A + B + C - math.sqrt(D))/(2*A)]
            
        else:
            
            x = [((A + B + C)/(2*A))]
                
        # Running through the solutions
        phi3_soln_arr = []
        
        for phi3soln in x:
            
            print(phi3soln)
            if phi3soln > 0 and phi3soln < 1:
                
                phi3_soln_arr.append(2*math.acos(math.sqrt(phi3soln)))
                
        # print('Solutions for phi3 are ' + str(phi3_soln_arr))
                
        # Checking if at least one solution exists for phi3
        if not phi3_soln_arr:
            
            return [[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]]
        
        else:
            
            for phi3 in phi3_soln_arr:
                
                # Obtaining the corresponding solution for phi2
                phi2 = math.pi + 2*math.atan2((Ul - mu)*math.tan(phi3/2), mu*math.sqrt(1 + Ul**2))
                
                # print('Solution for phi2 is ' + str(phi2))
                
                # Obtaining the solutions for phi1
                D = (1 - rR_mod**2)*(cos(phi3) - (1 - cos(phi2))*(rL_mod**2)*cos(phi3) - rL_mod*sin(phi2)*sin(phi3)) \
                    - rL_mod*rR_mod*math.sqrt(1 - rL_mod**2)*math.sqrt(1 - rR_mod**2)*(1 - cos(phi2))
                E = (1 - rR_mod**2)*(rL_mod*sin(phi2)*cos(phi3) + cos(phi2)*sin(phi3)) \
                    + rR_mod*math.sqrt(1 - rL_mod**2)*math.sqrt(1 - rR_mod**2)*sin(phi2)
                F = (1 - rR_mod**2)*alpha11 - rR_mod*math.sqrt(1 - rR_mod**2)*(alpha13 + alpha31)\
                    + (rR_mod**2)*alpha33 - rR_mod**2*(cos(phi2) + (1 - cos(phi2))*rL_mod**2) \
                    + rR_mod*math.sqrt(1 - rL_mod**2)*math.sqrt(1 - rR_mod**2)*\
                        (rL_mod*(1 - cos(phi2))*cos(phi3) + sin(phi2)*sin(phi3))
                
                if math.sqrt(D**2 + E**2) <= 10**(-6):
                    
                    continue # Since phi1 cannot be solved for using the equation obtained
                    
                else:
                    
                    gamma = math.atan2(E, D) # Computing the angle to be used to solve for phi1
                    
                    # Obtaining the RHS for solving for phi1
                    phi1_RHS = F/math.sqrt(D**2 + E**2)
                        
                    # Checking if phi1_RHS is in [-1, 1] with a tolerance
                    if abs(phi1_RHS) > 1 and abs(phi1_RHS) <= 1 + 10**(-6):
                        
                        phi1_RHS = np.sign(phi1_RHS)
                        
                    if abs(phi1_RHS) > 1:
                        
                        continue # Since solution for phi1 does not exist
                        
                    # Obtaining the solutions for phi1
                    phi1_soln_arr = [np.mod(math.acos(phi1_RHS) - gamma, 2*math.pi),\
                                     np.mod(2*math.pi - math.acos(phi1_RHS) - gamma, 2*math.pi)]
                        
                    # Checking if either solution is nearly equal to zero or 2*math.pi
                    if 2*math.pi - phi1_soln_arr[0] <= 10**(-6) or phi1_soln_arr[0] <= 10**(-6):
                        
                        phi1_soln_arr[0] = 0
                        
                    if 2*math.pi - phi1_soln_arr[1] <= 10**(-6) or phi1_soln_arr[1] <= 10**(-6):
                        
                        phi1_soln_arr[1] = 0
                        
                    # Checking if the two solutions are equal
                    if abs(phi1_soln_arr[1] - phi1_soln_arr[0]) <= 10**(-6):
                        
                        phi1_soln_arr = [np.mod(math.acos(phi1_RHS) - gamma, 2*math.pi)]
                        
                    # print('Solutions for phi1 are ' + str(phi1_soln_arr))
                        
                # Obtaining the solutions for solving for phi4
                G = rR_mod*math.sqrt(1 - rL_mod**2)*math.sqrt(1 - rR_mod**2)\
                    *(rL_mod*(1 - cos(phi2))*cos(phi3) + sin(phi2)*sin(phi3))\
                    + (1 - rR_mod**2)*(cos(phi2) + (1 - cos(phi2))*rL_mod**2)
                H = math.sqrt(1 - rL_mod**2)*math.sqrt(1 - rR_mod**2)*(-rL_mod*(1 - cos(phi2))*sin(phi3) + sin(phi2)*cos(phi3))
                I = alpha33 + rR_mod*math.sqrt(1 - rL_mod**2)*math.sqrt(1 - rR_mod**2)*\
                    (rL_mod*(1 - cos(phi2))*cos(phi3) + sin(phi2)*sin(phi3)) \
                    - rR_mod**2*(cos(phi2) + (1 - cos(phi2))*rL_mod**2)
                        
                if math.sqrt(G**2 + H**2) <= 10**(-6):
                    
                    continue # Since phi4 cannot be solved for using the equation obtained
                    
                else:
                    
                    beta = math.atan2(H, G)
                    
                    # Checking if the RHS for solving for phi4 is in [-1, 1] with a tolerance
                    phi4_RHS = I/math.sqrt(G**2 + H**2)
                    
                    if abs(phi4_RHS) > 1 and abs(phi4_RHS) <= 1 + 10**(-6):
                        
                        phi4_RHS = np.sign(phi4_RHS)
                        
                    if abs(phi4_RHS) > 1:
                        
                        continue # Since solution for phi4 does not exist
                        
                    # Obtaining the solutions for phi4
                    phi4_soln_arr = [np.mod(math.acos(phi4_RHS) + beta, 2*math.pi),\
                                     np.mod(2*math.pi - math.acos(phi4_RHS) + beta, 2*math.pi)]
                        
                    # Checking if either solution is nearly equal to zero or 2*math.pi
                    if 2*math.pi - phi4_soln_arr[0] <= 10**(-6) or phi4_soln_arr[0] <= 10**(-6):
                        
                        phi4_soln_arr[0] = 0
                        
                    if 2*math.pi - phi4_soln_arr[1] <= 10**(-6) or phi4_soln_arr[1] <= 10**(-6):
                        
                        phi4_soln_arr[1] = 0
                        
                    # Checking if the two solutions are equal
                    if abs(phi4_soln_arr[1] - phi4_soln_arr[0]) <= 10**(-6):
                        
                        phi4_soln_arr = [np.mod(math.acos(phi4_RHS) + beta, 2*math.pi)]
                        
                    # print('Solutions for phi4 are ' + str(phi4_soln_arr))
                        
                # Running through the solutions for phi1 and phi4, and checking if the final
                # configuration is reached
                for phi1 in phi1_soln_arr:
                    
                    for phi4 in phi4_soln_arr:
                        
                        # Obtaining the final configuration of the path
                        _, _, _, fin_config_path, _, _, _ =\
                            points_path(np.identity(3), rL_mod, rR_mod, 1, [phi1, phi2, phi3, phi4], path_type)
                            
                        # Checking if the minimum and maximum value in the difference in the final
                        # configurations is small
                        if abs(max(map(max, fin_config_path - fin_config_mod))) <= 10**(-6)\
                            and abs(min(map(min, fin_config_path - fin_config_mod))) <= 10**(-6):
                            
                            if path_type == 'glgr':    
                            
                                # Appending the path length, path parameters, and path cost
                                path_cost = R*(phi1 + phi3) + rL*(1 + muL*Ul)*phi2 + rR*(1 + muR*Ur)*phi4
                                path_length = R*(phi1 + phi3) + rL*phi2 + rR*phi4
                                
                            path_params.append([path_length, phi1, phi2, phi3, phi4, path_cost])
                            
        # Checking if no solution was obtained for the path
        if len(path_params) == 0:
            
            print(path_type.upper() + ' path does not exist.')
            path_params.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
    
    return path_params

def path_generation_sphere_five_seg(ini_config, fin_config, rL, rR, R, muL, muR, path_type = 'lgrgl'):
    
    # Modifying the configurations and the parameters of the turn
    fin_config_mod = modifying_initial_final_configurations_unit_sphere(ini_config, fin_config, R)
    rL_mod = rL/R
    rR_mod = rR/R
    
    Ul = math.sqrt(((R/rL)**2 - 1))
    Ur = math.sqrt(((R/rR)**2 - 1))
    
    path_types = ['lgrgl', 'grglg', 'rglgr', 'glgrg']
    
    if path_type not in path_types:
        
        raise Exception('Incorrect path type passed.')
        
    # Path construction for muL = muR, and muL <= Ul, muR <= Ur
    if muL != muR or muL > Ul or muR > Ur:
        
        raise Exception('Conditions for path generation are not met.')
        
    # Defining variables corresponding to the final configuration
    alpha11 = fin_config_mod[0, 0]; alpha12 = fin_config_mod[0, 1]; alpha13 = fin_config_mod[0, 2];
    alpha21 = fin_config_mod[1, 0]; alpha22 = fin_config_mod[1, 1]; alpha23 = fin_config_mod[1, 2];
    alpha31 = fin_config_mod[2, 0]; alpha32 = fin_config_mod[2, 1]; alpha33 = fin_config_mod[2, 2];
    
    # Storing the details of the path
    path_params = []
    
    mu = muL
    
    if path_type == 'lgrgl':
        
        # Constructing the equation used to solve for the angle of the G segments and the angle
        # of the R segment
        A = 8*(-1 + rL_mod**2)*(2*Ur*mu*(-1 + rR_mod*math.sqrt(1 + Ur**2)) \
                                + mu**2*(1 + rR_mod**2 - 2*rR_mod*math.sqrt(1 + Ur**2))\
                                + Ur**2*(1 + rR_mod**2*mu**2))
        B = -8*Ur*mu*(4 - 3*rR_mod*math.sqrt(1 + Ur**2) \
                      + rL_mod*math.sqrt((1 - rL_mod**2)*(1 - rR_mod**2)*(1 + Ur**2))\
                      + rL_mod**2*(-4 + 3*rR_mod*math.sqrt(1 + Ur**2))) \
            -8*mu**2*(-2 - rR_mod**2 + 3*rR_mod*math.sqrt(1 + Ur**2) \
                      + rL_mod*math.sqrt(1 - rL_mod**2)*math.sqrt(1 - rR_mod**2)*(rR_mod - math.sqrt(1 + Ur**2)) \
                      + rL_mod**2*(2 + rR_mod**2 - 3*rR_mod*math.sqrt(1 + Ur**2))) \
            -8*Ur**2*(-2 + rL_mod*rR_mod*math.sqrt(1 - rL_mod**2)*math.sqrt(1 - rR_mod**2)*mu**2 \
                      - rR_mod**2*mu**2 + rL_mod**2*(2 + rR_mod**2*mu**2))
        C = 2*Ur*mu*(9 - 4*rR_mod*math.sqrt(1 + Ur**2) - alpha11 \
                     + rL_mod*math.sqrt(1 - rL_mod**2)*(4*math.sqrt(1 - rR_mod**2)*math.sqrt(1 + Ur**2)\
                                                        - alpha13 - alpha31) \
                     + rL_mod**2*(-8 + 4*rR_mod*math.sqrt(1 + Ur**2) + alpha11 - alpha33)) \
            + 2*mu**2*(-4 - rR_mod**2 + 4*rR_mod*math.sqrt(1 + Ur**2) \
                       + 2*rL_mod*math.sqrt(1 - rL_mod**2)*math.sqrt(1 - rR_mod**2)*(rR_mod - 2*math.sqrt(1 + Ur**2))\
                       + rL_mod**2*(3 + 2*rR_mod**2 - 4*rR_mod*math.sqrt(1 + Ur**2))) \
            + Ur**2*(-9 + (1 - mu**2)*alpha11 + (1 - 2*rR_mod**2)*mu**2 \
                     + rL_mod*math.sqrt(1 - rL_mod**2)*((alpha13 + alpha31)*(1 - mu**2) \
                                                        + 4*rR_mod*math.sqrt(1 - rR_mod**2)*mu**2) \
                     + rL_mod**2*(8 + (alpha33 - alpha11)*(1 - mu**2) - 2*mu**2 + 4*rR_mod**2*mu**2))
        D = (1 + (rL_mod**2 - 1)*alpha11 - rL_mod*math.sqrt(1 - rL_mod**2)*(alpha13 + alpha31) \
             - (rL_mod**2)*alpha33)*(Ur - mu)**2
        
        # coeff = [D, C, B, A]
        coeff = [A, B, C, D]
        
        # # Obtaining the roots of the cubic polynomial
        # x = poly.polyroots(coeff)
        x = np.roots(coeff)
        # print(x)
        x_real = x.real[abs(x.imag) < 1e-5]
        # print(x_real)
        
        phi2_soln_arr = []
        
        # Running through each solution in x
        for soln in x_real:
            
            # Checking if the value is in (0, 1)
            if soln > 0 and soln < 1:
                
                phi2_soln_arr.append(2*math.acos(math.sqrt(soln)))
                
        # print(phi2_soln_arr)
                
        # Running through each solution for phi2
        for phi2 in phi2_soln_arr:
            
            # Obtaining the corresponding solution for phi3
            phi3 = math.pi + 2*math.atan2((Ur - mu)*math.tan(phi2/2), mu*math.sqrt(1 + Ur**2))
            
            # print('phi3 is ' + str(phi3))
            
            # Computing the solutions for phi1 and phi4
            beta11 = (cos(phi2))**2*(1 - rR_mod**2 + rR_mod**2*cos(phi3)) - cos(phi3)*(sin(phi2))**2 \
                - rR_mod*sin(2*phi2)*sin(phi3)
            beta13 = math.sqrt(1 - rR_mod**2)*(rR_mod*cos(phi2)*(-1 + cos(phi3)) - sin(phi2)*sin(phi3))
            beta33 = rR_mod**2 + (1 - rR_mod**2)*cos(phi3)
            beta12 = -cos(phi2)*(1 - rR_mod**2 + (1 + rR_mod**2)*cos(phi3))*sin(phi2) \
                - rR_mod*sin(phi3)*cos(2*phi2)
            beta23 = math.sqrt(1 - rR_mod**2)*(rR_mod*sin(phi2)*(-1 + cos(phi3)) + cos(phi2)*sin(phi3))
            
            D = rL_mod*math.sqrt(1 - rL_mod**2)*(beta33 - beta11) + (1 - 2*rL_mod**2)*beta13
            E = -math.sqrt(1 - rL_mod**2)*beta12 + rL_mod*beta23
            F = (-alpha11 + rL_mod**2*(alpha11 + alpha33) + rL_mod*math.sqrt(1 - rL_mod**2)*(alpha31 - alpha13)\
                 - (2*rL_mod**2 - 1)*((1 - rL_mod**2)*beta11 + 2*rL_mod*math.sqrt(1 - rL_mod**2)*beta13 + rL_mod**2*beta33))\
                /(2*rL_mod*math.sqrt(1 - rL_mod**2))
            G = (-alpha11 + rL_mod**2*(alpha11 + alpha33) + rL_mod*math.sqrt(1 - rL_mod**2)*(alpha13 - alpha31)\
                 - (2*rL_mod**2 - 1)*((1 - rL_mod**2)*beta11 + 2*rL_mod*math.sqrt(1 - rL_mod**2)*beta13 + rL_mod**2*beta33))\
                /(2*rL_mod*math.sqrt(1 - rL_mod**2))
                
            if math.sqrt(D**2 + E**2) <= 10**(-6):
                    
                continue # Since phi1 and phi44 cannot be solved for using the equation obtained
                    
            else:
                
                gamma = math.atan2(E, D)
                
                phi1_RHS = F/math.sqrt(D**2 + E**2)
                
                # Checking if phi1_RHS is in [-1, 1] with a tolerance
                if abs(phi1_RHS) > 1 and abs(phi1_RHS) <= 1 + 10**(-6):
                    
                    phi1_RHS = np.sign(phi1_RHS)
                    
                if abs(phi1_RHS) > 1:
                    
                    continue # Since solution for phi1 does not exist
                
                # Obtaining the solutions for phi1
                phi1_soln_arr = [np.mod(math.acos(phi1_RHS) + gamma, 2*math.pi),\
                                 np.mod(2*math.pi - math.acos(phi1_RHS) + gamma, 2*math.pi)]
                    
                # Checking if either solution is nearly equal to zero or 2*math.pi
                if 2*math.pi - phi1_soln_arr[0] <= 10**(-6) or phi1_soln_arr[0] <= 10**(-6):
                    
                    phi1_soln_arr[0] = 0
                    
                if 2*math.pi - phi1_soln_arr[1] <= 10**(-6) or phi1_soln_arr[1] <= 10**(-6):
                    
                    phi1_soln_arr[1] = 0
                    
                # Checking if the two solutions are equal
                if abs(phi1_soln_arr[1] - phi1_soln_arr[0]) <= 10**(-6):
                    
                    phi1_soln_arr = [np.mod(math.acos(phi1_RHS) + gamma, 2*math.pi)]
                    
                # print('Solutions for phi1 are ' + str(phi1_soln_arr))
                
                # Checking if the RHS for solving for phi4 is in [-1, 1] with a tolerance
                phi4_RHS = G/math.sqrt(D**2 + E**2)
                
                if abs(phi4_RHS) > 1 and abs(phi4_RHS) <= 1 + 10**(-6):
                    
                    phi4_RHS = np.sign(phi4_RHS)
                    
                if abs(phi4_RHS) > 1:
                    
                    continue # Since solution for phi4 does not exist
                    
                # Obtaining the solutions for phi4
                phi4_soln_arr = [np.mod(math.acos(phi4_RHS) + gamma, 2*math.pi),\
                                 np.mod(2*math.pi - math.acos(phi4_RHS) + gamma, 2*math.pi)]
                    
                # Checking if either solution is nearly equal to zero or 2*math.pi
                if 2*math.pi - phi4_soln_arr[0] <= 10**(-6) or phi4_soln_arr[0] <= 10**(-6):
                    
                    phi4_soln_arr[0] = 0
                    
                if 2*math.pi - phi4_soln_arr[1] <= 10**(-6) or phi4_soln_arr[1] <= 10**(-6):
                    
                    phi4_soln_arr[1] = 0
                    
                # Checking if the two solutions are equal
                if abs(phi4_soln_arr[1] - phi4_soln_arr[0]) <= 10**(-6):
                    
                    phi4_soln_arr = [np.mod(math.acos(phi4_RHS) + gamma, 2*math.pi)]
                    
                # print('Solutions for phi4 are ' + str(phi4_soln_arr))
                
                # Running through the solutions for phi1 and phi4, and checking if the final
                # configuration is reached
                for phi1 in phi1_soln_arr:
                    
                    for phi4 in phi4_soln_arr:
                        
                        # Obtaining the final configuration of the path
                        _, _, _, fin_config_path, _, _, _ =\
                            points_path(np.identity(3), rL_mod, rR_mod, 1,\
                                        [phi1, phi2, phi3, phi2, phi4], path_type)
                            
                        # Checking if the minimum and maximum value in the difference in the final
                        # configurations is small
                        if abs(max(map(max, fin_config_path - fin_config_mod))) <= 10**(-6)\
                            and abs(min(map(min, fin_config_path - fin_config_mod))) <= 10**(-6): 
                            
                            # Appending the path length, path parameters, and path cost
                            path_cost = 2*R*phi2 + rL*(1 + muL*Ul)*(phi1 + phi4) + rR*(1 + muR*Ur)*phi3
                            path_length = 2*R*phi2 + rL*(phi1 + phi4) + rR*phi3
                                
                            path_params.append([path_length, phi1, phi2, phi3, phi2, phi4, path_cost])
                            
    elif path_type == 'glgrg':
        
        # Constructing the equation to solve for the middle G segment and the L and R segments
        A = -8*(Ul*Ur*mu)**2
        B = 2*(-1 + rL_mod**2)*mu**2*(2*Ur*mu + 2*(-1 + rR_mod**2)*mu**2 + Ur**2*(-1 + (-1 + 2*rR_mod**2)*mu**2))\
            - 2*Ul*mu*(2*Ur*(-1 + alpha33)*mu - 2*(-1 + rR_mod**2)*mu**2 + Ur**2*(1 + (3 - 2*rR_mod**2)*mu**2 + alpha33*(-1 + mu**2)))\
            + Ul**2*(2*(-1 + rR_mod**2)*(-1 + (-1 + 2*rL_mod**2)*mu**2)*mu**2 - 2*Ur*mu*(1 - alpha33 + (3 - 2*rL_mod**2 + alpha33)*mu**2)\
                     - Ur**2*(-1 + 2*(-7 + rL_mod**2 + rR**2)*mu**2 - (-1 + 2*rL_mod**2)*(-1 + 2*rR_mod**2)*mu**4 + alpha33*(-1 + mu**2)**2))
        C = mu**2*(-2*Ur*(-3 + 2*rL_mod**2 + alpha33)*mu + 2*(-2 + rL_mod**2 + rR_mod**2)*mu**2 + Ur**2*(-3 + 2*rL_mod**2 + (1 - mu**2)*alpha33 + (2*rR_mod**2 - 1)*mu**2))\
            + Ul**2*(mu**2*(-3 + 2*rR_mod**2 + (1 - mu**2)*alpha33 + (2*rL_mod**2 - 1)*mu**2) + 2*Ur**2*(-1 + (1 - mu**2)*alpha33 + (-3 + rL_mod**2 + rR_mod**2)*mu**2)\
                     + 2*Ur*mu*(2 + (3 - 2*rL_mod**2)*mu**2 + alpha33*(-2 + mu**2)))\
            + 2*Ul*mu*(-(-3 + 2*rR_mod**2 + alpha33)*mu**2 - 2*Ur*mu*(2*(1 - alpha33) + mu**2) + Ur**2*(2 + (3 - 2*rR**2)*mu**2 + (-2 + mu**2)*alpha33))
        D = (1 - alpha33)*((Ul - mu)*(Ur - mu))**2
    
        coeff = [A, B, C, D]
        
        # # Obtaining the roots of the cubic polynomial
        x = np.roots(coeff)
        x_real = x.real[abs(x.imag) < 1e-5]
        # print(x_real)
        
        phi3_soln_arr = []
        
        # Running through each solution in x
        for soln in x_real:
            
            # Checking if the value is in (0, 1)
            if soln > 0 and soln < 1:
                
                phi3_soln_arr.append(2*math.acos(math.sqrt(soln)))
                
        # print(phi3_soln_arr)
        
        # Running through each solution for phi3
        for phi3 in phi3_soln_arr:
            
            # Computing the solution for phi2 and phi4
            phi2 = math.pi + 2*math.atan2((Ul - mu)*math.tan(phi3/2), mu*math.sqrt(1 + Ul**2))
            phi4 = math.pi + 2*math.atan2((Ur - mu)*math.tan(phi3/2), mu*math.sqrt(1 + Ur**2))
            
            # Obtaining the solutions for phi1
            E = - math.sqrt(1 - rR_mod**2)*(1 - (1 - cos(phi2))*rL_mod**2)*(rR_mod*cos(phi3)*(1 - cos(phi4)) + sin(phi3)*sin(phi4))\
                - rL_mod*math.sqrt(1 - rR_mod**2)*sin(phi2)*(- rR_mod*sin(phi3)*(1 - cos(phi4)) + cos(phi3)*sin(phi4))\
                + rL_mod*math.sqrt(1 - rL_mod**2)*(1 - cos(phi2))*(cos(phi4) + (1 - cos(phi4))*rR_mod**2)
            F = - rL_mod*math.sqrt(1 - rR_mod**2)*sin(phi2)*(rR_mod*cos(phi3)*(1 - cos(phi4)) + sin(phi3)*sin(phi4))\
                + math.sqrt(1 - rR_mod**2)*cos(phi2)*(- rR_mod*sin(phi3)*(1 - cos(phi4)) + cos(phi3)*sin(phi4))\
                - math.sqrt(1 - rL_mod**2)*sin(phi2)*(cos(phi4) + (1 - cos(phi4))*rR_mod**2)
            G = - rL*math.sqrt(1 - rL_mod**2)*math.sqrt(1 - rR_mod**2)*(1 - cos(phi2))*(rR_mod*cos(phi3)*(1 - cos(phi4)) + sin(phi3)*sin(phi4))\
                + math.sqrt(1 - rL_mod**2)*math.sqrt(1 - rR_mod**2)*sin(phi2)*(- rR_mod*sin(phi3)*(1 - cos(phi4)) + cos(phi3)*sin(phi4))\
                + (cos(phi2) + (1 - cos(phi2))*rL_mod**2)*(cos(phi4) + (1 - cos(phi4))*rR_mod**2)
                
            if math.sqrt(E**2 + F**2) <= 10**(-6):
                    
                continue # Since phi1 cannot be solved for using the equation obtained
                    
            else:
                
                gamma = math.atan2(F, E)
                
                phi1_RHS = (math.sqrt(1 - rL_mod**2)*alpha13 + rL_mod*(alpha33 - G))/(math.sqrt(1 - rL_mod**2)*math.sqrt(E**2 + F**2))
                
                # Checking if phi1_RHS is in [-1, 1] with a tolerance
                if abs(phi1_RHS) > 1 and abs(phi1_RHS) <= 1 + 10**(-6):
                    
                    phi1_RHS = np.sign(phi1_RHS)
                    
                if abs(phi1_RHS) > 1:
                    
                    continue # Since solution for phi1 does not exist
                
                # Obtaining the solutions for phi1
                phi1_soln_arr = [np.mod(math.acos(phi1_RHS) - gamma, 2*math.pi),\
                                 np.mod(2*math.pi - math.acos(phi1_RHS) - gamma, 2*math.pi)]
                    
                # Checking if either solution is nearly equal to zero or 2*math.pi
                if 2*math.pi - phi1_soln_arr[0] <= 10**(-6) or phi1_soln_arr[0] <= 10**(-6):
                    
                    phi1_soln_arr[0] = 0
                    
                if 2*math.pi - phi1_soln_arr[1] <= 10**(-6) or phi1_soln_arr[1] <= 10**(-6):
                    
                    phi1_soln_arr[1] = 0
                    
                # Checking if the two solutions are equal
                if abs(phi1_soln_arr[1] - phi1_soln_arr[0]) <= 10**(-6):
                    
                    phi1_soln_arr = [np.mod(math.acos(phi1_RHS) - gamma, 2*math.pi)]
                    
                # print('Solutions for phi1 are ' + str(phi1_soln_arr))
                
            # Obtaining the solutions for phi5
            H = sqrt(1 - rL_mod**2)*(rL_mod*(1 - cos(phi2))*cos(phi3) + sin(phi2)*sin(phi3))*(1 - (1 - cos(phi4))*rR_mod**2)\
                + rR_mod*sqrt(1 - rL_mod**2)*(-rL_mod*(1 - cos(phi2))*sin(phi3) + sin(phi2)*cos(phi3))*sin(phi4)\
                - rR_mod*sqrt(1 - rR_mod**2)*(cos(phi2) + (1 - cos(phi2))*rL_mod**2)*(1 - cos(phi4))
            I = - rR_mod*sqrt(1 - rL_mod**2)*(rL_mod*(1 - cos(phi2))*cos(phi3) + sin(phi2)*sin(phi3))*sin(phi4)\
                + sqrt(1 - rL_mod**2)*(-rL_mod*(1 - cos(phi2))*sin(phi3) + sin(phi2)*cos(phi3))*cos(phi4)\
                - sqrt(1 - rR_mod**2)*(cos(phi2) + (1 - cos(phi2))*rL_mod**2)*sin(phi4)
            J = - rR_mod*sqrt(1 - rL_mod**2)*sqrt(1 - rR_mod**2)*(rL_mod*(1 - cos(phi2))*cos(phi3) + sin(phi2)*sin(phi3))*(1 - cos(phi4))\
                + sqrt(1 - rL_mod**2)*sqrt(1 - rR_mod**2)*(-rL_mod*(1 - cos(phi2))*sin(phi3) + sin(phi2)*cos(phi3))*sin(phi4)\
                + (cos(phi2) + (1 - cos(phi2))*rL_mod**2)*(cos(phi4) + (1 - cos(phi4))*rR_mod**2)
                
            if math.sqrt(H**2 + I**2) <= 10**(-6):
                        
                # print('phi5 cannot be solved for.')
                continue # Since phi5 cannot be solved for using the equation obtained
                    
            else:
                
                beta = math.atan2(I, H)
                
                phi5_RHS = (sqrt(1 - rL_mod**2)*alpha31 + rL_mod*(alpha33 - J))/(sqrt(1 - rL_mod**2)*math.sqrt(H**2 + I**2))
                
                # Checking if phi5_RHS is in [-1, 1] with a tolerance
                if abs(phi5_RHS) > 1 and abs(phi5_RHS) <= 1 + 10**(-6):
                    
                    phi5_RHS = np.sign(phi5_RHS)
                    
                if abs(phi5_RHS) > 1:
                    
                    continue # Since solution for phi5 does not exist
                
                # Obtaining the solutions for phi5
                phi5_soln_arr = [np.mod(math.acos(phi5_RHS) + beta, 2*math.pi),\
                                 np.mod(2*math.pi - math.acos(phi5_RHS) + beta, 2*math.pi)]
                    
                # Checking if either solution is nearly equal to zero or 2*math.pi
                if 2*math.pi - phi5_soln_arr[0] <= 10**(-6) or phi5_soln_arr[0] <= 10**(-6):
                    
                    phi5_soln_arr[0] = 0
                    
                if 2*math.pi - phi5_soln_arr[1] <= 10**(-6) or phi5_soln_arr[1] <= 10**(-6):
                    
                    phi5_soln_arr[1] = 0
                    
                # Checking if the two solutions are equal
                if abs(phi5_soln_arr[1] - phi5_soln_arr[0]) <= 10**(-6):
                    
                    phi5_soln_arr = [np.mod(math.acos(phi5_RHS) + beta, 2*math.pi)]
                    
                # print('Solutions for phi5 are ' + str(phi5_soln_arr))
                
            # Running through the solutions for phi1 and phi4, and checking if the final
            # configuration is reached            
            for phi1 in phi1_soln_arr:
                
                for phi5 in phi5_soln_arr:
                    
                    # Obtaining the final configuration of the path
                    _, _, _, fin_config_path, _, _, _ =\
                        points_path(np.identity(3), rL_mod, rR_mod, 1,\
                                    [phi1, phi2, phi3, phi4, phi5], path_type)
                        
                    # Checking if the minimum and maximum value in the difference in the final
                    # configurations is small                    
                    if abs(max(map(max, fin_config_path - fin_config_mod))) <= 10**(-6)\
                        and abs(min(map(min, fin_config_path - fin_config_mod))) <= 10**(-6):
                        
                        # Appending the path length, path parameters, and path cost
                        path_cost = R*(phi1 + phi3 + phi5) + rL*(1 + muL*Ul)*phi2 + rR*(1 + muR*Ur)*phi4
                        path_length = R*(phi1 + phi3 + phi5) + rL*phi2 + rR*phi4
                            
                        path_params.append([path_length, phi1, phi2, phi3, phi4, phi5, path_cost])
                
    # Checking if no solution was obtained for the path
    if len(path_params) == 0:
        
        print(path_type.upper() + ' path does not exist.')
        path_params.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
                
    return path_params

def path_generation_sphere_six_seg(ini_config, fin_config, rL, rR, R, muL, muR, path_type = 'glgrgl'):
    
    # Modifying the configurations and the parameters of the turn
    fin_config_mod = modifying_initial_final_configurations_unit_sphere(ini_config, fin_config, R)
    rL_mod = rL/R
    rR_mod = rR/R
    
    Ul = math.sqrt(((R/rL)**2 - 1))
    Ur = math.sqrt(((R/rR)**2 - 1))
    
    path_types = ['lgrglg', 'glgrgl']
    
    if path_type not in path_types:
        
        raise Exception('Incorrect path type passed.')
        
    # Path construction for muL = muR, and muL <= Ul, muR <= Ur
    if muL != muR or muL > Ul or muR > Ur:
        
        raise Exception('Conditions for path generation are not met.')
        
    # Defining variables corresponding to the final configuration
    alpha11 = fin_config_mod[0, 0]; alpha12 = fin_config_mod[0, 1]; alpha13 = fin_config_mod[0, 2];
    alpha21 = fin_config_mod[1, 0]; alpha22 = fin_config_mod[1, 1]; alpha23 = fin_config_mod[1, 2];
    alpha31 = fin_config_mod[2, 0]; alpha32 = fin_config_mod[2, 1]; alpha33 = fin_config_mod[2, 2];
    
    # Storing the details of the path
    path_params = []
    
    mu = muL
    
    if path_type == 'glgrgl':
        
        # Constructing the equation to solve for the middle G segment and the L and R segments
        A = -16*sqrt(1 - rL_mod**2)*Ul**2*Ur**2*mu
        B = 4*sqrt(1 - rL_mod**2)*Ur*mu*(-6*Ul*Ur*mu + 2*(1 - rL_mod**2)*Ur*mu**2\
                                         + Ul**2*(-4*mu + Ur*(11 + (1 - 2*rL_mod**2)*mu**2)))
        C = 2*mu**2*rL_mod**3*(Ul**2 + 1)*(2*mu**2*(rR_mod**2 - 1) + Ur**2*(mu**2*(2*rR_mod**2 - 1) - 1) + 2*mu*Ur)\
            - sqrt(1 - rL_mod**2)*(Ul**2*(4*mu**3 + Ur**2*(alpha31*(mu**2 - 1)**2 + 4*mu*(mu**2 + 10))\
                                          - 2*mu*Ur*(alpha31*(1 - mu**2) + mu**3 + 15*mu)) \
                                   + 2*mu*Ul*Ur*(2*mu*(alpha31 + 4*mu) + alpha31*(mu**2 - 1)*Ur\
                                                 - mu*(mu**2 + 21)*Ur) + 4*mu**3*Ur*(4*Ur - mu))\
            + 4*mu**3*sqrt(1 - rL_mod**2)*rL_mod**2*(Ul**2 + 1)*Ur*(2*Ur - mu)\
            - rL_mod*(Ul**2*(2*mu**2*(mu**2 + 1)*(rR_mod**2 - 1)\
                             + Ur**2*(alpha33*(mu**2 - 1)**2 + (mu**2 + 1)*(mu**2*(2*rR_mod**2 - 1) - 1)) \
                             + 2*Ur*mu*((alpha33 + 1)*mu**2 + 1 - alpha33))\
                      + 2*mu*Ul*(-2*mu**2*(rR_mod**2 - 1) + 2*(alpha33 - 1)*mu*Ur\
                                 + Ur**2*(alpha33*(mu**2 - 1) + mu**2*(1 - 2*rR_mod**2) + 1))\
                      + 2*mu**2*(2*mu**2*(rR_mod**2 - 1) + Ur**2*(mu**2*(2*rR_mod**2 - 1) - 1) + 2*mu*Ur))
        D = 2*mu**2*rL_mod**3*(Ul**2 + 1)*(Ur - mu)**2\
            + sqrt(1 - rL_mod**2)*(Ul**2*(mu**2*(alpha31*(1 - mu**2) + 4*mu) \
                                          + 2*Ur**2*(alpha31*(1 - mu**2) + 6*mu)\
                                          + 2*mu*Ur*(alpha31*(mu**2 - 2) - 7*mu)) \
                                   - 2*mu*Ul*(mu**2*(alpha31 + mu) + Ur**2*(alpha31*(2 - mu**2) + 9*mu)\
                                              - 4*mu*Ur*(alpha31 + 2*mu))\
                                   + mu**2*Ur*(Ur*(alpha31*(1 - mu**2) + 8*mu) - 2*mu*(alpha31 + 3*mu)))\
            - rL_mod*(Ul**2*(mu**2*(alpha33*(mu**2 - 1) + mu**2 - 2*rR_mod**2 + 3)\
                             + 2*Ur**2*(alpha33*(mu**2 - 1) - mu**2*(rR_mod**2 - 1) + 1) \
                             - 2*mu*Ur*(alpha33*(mu**2 - 2) + mu**2 + 2))\
                      - 2*mu*Ul*(-mu**2*(alpha33 + 2*rR_mod**2 - 3)\
                                 + Ur**2*(alpha33*(mu**2 - 2) + mu**2*(1 - 2*rR_mod**2) + 2) + 4*(alpha33 - 1)*mu*Ur)\
                      + mu**2*(-2*mu**2*(rR_mod**2 - 2) + 2*(alpha33 - 3)*mu*Ur\
                               + Ur**2*(alpha33*(mu**2 - 1) + mu**2*(1 - 2*rR_mod**2) + 3)))
        E = -(sqrt(1 - rL_mod**2)*alpha31 + rL_mod*(-1 + alpha33))*(Ul - mu)**2*(Ur - mu)**2
        
        coeff = [A, B, C, D, E]
        
        # # Obtaining the roots of the cubic polynomial
        x = np.roots(coeff)
        x_real = x.real[abs(x.imag) < 1e-5]
        # print(x_real)
        
        phi3_soln_arr = []
        
        # Running through each solution in x
        for soln in x_real:
            
            # Checking if the value is in (0, 1)
            if soln > 0 and soln < 1:
                
                phi3_soln_arr.append(2*math.acos(math.sqrt(soln)))
                
        # print(phi3_soln_arr)
        
        # Running through each solution for phi3
        for phi3 in phi3_soln_arr:
            
            # Computing the solution for phi2 and phi4
            phi2 = math.pi + 2*math.atan2((Ul - mu)*math.tan(phi3/2), mu*math.sqrt(1 + Ul**2))
            phi4 = math.pi + 2*math.atan2((Ur - mu)*math.tan(phi3/2), mu*math.sqrt(1 + Ur**2))
            # print('Solution for phi2 is ' + str(phi2))
            # print('Solution for phi4 is ' + str(phi4))
            
            # Obtaining the solutions for phi1
            F = sqrt(1 - rL_mod**2)*((1 - (1 - cos(phi4))*rR_mod**2)*(cos(phi3))**2\
                                     - rR_mod*sin(2*phi3)*sin(phi4) - (sin(phi3))**2*cos(phi4))\
                - rL_mod*sqrt(1 - rR_mod**2)*(rR_mod*(1 - cos(phi4))*cos(phi3) + sin(phi3)*sin(phi4))
            G = sqrt(1 - rL_mod**2)*((1 - rR_mod**2 + (1 + rR_mod**2)*cos(phi4))*sin(phi3)*cos(phi3)\
                                     + rR_mod*cos(2*phi3)*sin(phi4))\
                + rL_mod*sqrt(1 - rR_mod**2)*(cos(phi3)*sin(phi4) - rR_mod*sin(phi3)*(1 - cos(phi4)))
            H = - sqrt(1 - rL_mod**2)*sqrt(1 - rR_mod**2)*(rR_mod*(1 - cos(phi4))*cos(phi3) + sin(phi3)*sin(phi4))\
                + rL_mod*(cos(phi4) + (1 - cos(phi4))*rR_mod**2)
            I = (1 - (1 - cos(phi2))*rL_mod**2)*F - rL_mod*sin(phi2)*G + (1 - cos(phi2))*rL_mod*sqrt(1 - rL_mod**2)*H
            J = rL_mod*sin(phi2)*F + cos(phi2)*G - sin(phi2)*sqrt(1 - rL_mod**2)*H
            K = (1 - cos(phi2))*rL_mod*sqrt(1 - rL_mod**2)*F \
                + sin(phi2)*sqrt(1 - rL_mod**2)*G + (cos(phi2) + (1 - cos(phi2))*rL_mod**2)*H
                
            if math.sqrt(I**2 + J**2) <= 10**(-6):
                    
                continue # Since phi1 cannot be solved for using the equation obtained
                    
            else:
                
                gamma = math.atan2(J, I)
                
                phi1_RHS = (alpha11 + rL_mod*math.sqrt(1 - rL_mod**2)*(alpha13 + alpha31)\
                            + rL_mod**2*(alpha33 - alpha11) - rL_mod*K)/(math.sqrt(1 - rL_mod**2)*math.sqrt(I**2 + J**2))
                
                # Checking if phi1_RHS is in [-1, 1] with a tolerance
                if abs(phi1_RHS) > 1 and abs(phi1_RHS) <= 1 + 10**(-6):
                    
                    phi1_RHS = np.sign(phi1_RHS)
                    
                if abs(phi1_RHS) > 1:
                    
                    continue # Since solution for phi1 does not exist
                
                # Obtaining the solutions for phi1
                phi1_soln_arr = [np.mod(math.acos(phi1_RHS) - gamma, 2*math.pi),\
                                 np.mod(2*math.pi - math.acos(phi1_RHS) - gamma, 2*math.pi)]
                    
                # Checking if either solution is nearly equal to zero or 2*math.pi
                if 2*math.pi - phi1_soln_arr[0] <= 10**(-6) or phi1_soln_arr[0] <= 10**(-6):
                    
                    phi1_soln_arr[0] = 0
                    
                if 2*math.pi - phi1_soln_arr[1] <= 10**(-6) or phi1_soln_arr[1] <= 10**(-6):
                    
                    phi1_soln_arr[1] = 0
                    
                # Checking if the two solutions are equal
                if abs(phi1_soln_arr[1] - phi1_soln_arr[0]) <= 10**(-6):
                    
                    phi1_soln_arr = [np.mod(math.acos(phi1_RHS) - gamma, 2*math.pi)]
                    
                # print('Solutions for phi1 are ' + str(phi1_soln_arr))
                
            # Obtaining the solutions for phi5
            beta11 = (cos(phi3))**2*(1 - rR_mod**2 + rR_mod**2*cos(phi4)) - cos(phi4)*(sin(phi3))**2 - rR_mod*sin(2*phi3)*sin(phi4)
            beta12 = -cos(phi3)*(1 - rR_mod**2 + (1 + rR_mod**2)*cos(phi4))*sin(phi3) - rR_mod*sin(phi4)*cos(2*phi3)
            beta13 = sqrt(1 - rR_mod**2)*(rR_mod*cos(phi3)*(-1 + cos(phi4)) - sin(phi3)*sin(phi4))
            beta22 = (cos(phi3))**2*cos(phi4) + (-1 + rR_mod**2 - rR_mod**2*cos(phi4))*(sin(phi3))**2 - rR_mod*sin(2*phi3)*sin(phi4)
            beta23 = sqrt(1 - rR_mod**2)*(rR_mod*sin(phi3)*(-1 + cos(phi4)) + cos(phi3)*sin(phi4))
            beta33 = rR_mod**2 + (1 - rR_mod**2)*cos(phi4)
            
            L = - rL_mod*(1 - rL_mod**2)*(rL_mod*(1 - cos(phi2))*beta11 - sin(phi2)*beta12)\
                + rL_mod*sqrt(1 - rL_mod**2)*(- cos(phi2) + (1 - 2*rL_mod**2)*(1 - cos(phi2)))*beta13\
                + (1 - rL_mod**2)**(1.5)*sin(phi2)*beta23 + (1 - rL_mod**2)*(cos(phi2) + (1 - cos(phi2))*rL_mod**2)*beta33
            M = (1 - rL_mod**2)*((1 - cos(phi2))*rL_mod*beta12 + sin(phi2)*beta22)\
                - sqrt(1 - rL_mod**2)*(cos(phi2) + rL_mod**2*(1 - cos(phi2)))*beta23
            N = rL_mod**2*(1 - rL_mod**2)*(1 - cos(phi2))*beta11 - rL_mod*(1 - rL_mod**2)*sin(phi2)*beta12\
                + rL_mod*sqrt(1 - rL_mod**2)*(cos(phi2) + 2*rL_mod**2*(1 - cos(phi2)))*beta13\
                + rL_mod**2*sqrt(1 - rL_mod**2)*sin(phi2)*beta23 + rL_mod**2*(cos(phi2) + (1 - cos(phi2))*rL_mod**2)*beta33
                
            if math.sqrt(L**2 + M**2) <= 10**(-6):
                        
                # print('phi5 cannot be solved for.')
                continue # Since phi5 cannot be solved for using the equation obtained
                    
            else:
                
                beta = math.atan2(M, L)
                
                phi5_RHS = (alpha33 - N)/sqrt(L**2 + M**2)
                
                # Checking if phi5_RHS is in [-1, 1] with a tolerance
                if abs(phi5_RHS) > 1 and abs(phi5_RHS) <= 1 + 10**(-6):
                    
                    phi5_RHS = np.sign(phi5_RHS)
                    
                if abs(phi5_RHS) > 1:
                    
                    continue # Since solution for phi5 does not exist
                
                # Obtaining the solutions for phi5
                phi5_soln_arr = [np.mod(math.acos(phi5_RHS) - beta, 2*math.pi),\
                                 np.mod(2*math.pi - math.acos(phi5_RHS) - beta, 2*math.pi)]
                    
                # Checking if either solution is nearly equal to zero or 2*math.pi
                if 2*math.pi - phi5_soln_arr[0] <= 10**(-6) or phi5_soln_arr[0] <= 10**(-6):
                    
                    phi5_soln_arr[0] = 0
                    
                if 2*math.pi - phi5_soln_arr[1] <= 10**(-6) or phi5_soln_arr[1] <= 10**(-6):
                    
                    phi5_soln_arr[1] = 0
                    
                # Checking if the two solutions are equal
                if abs(phi5_soln_arr[1] - phi5_soln_arr[0]) <= 10**(-6):
                    
                    phi5_soln_arr = [np.mod(math.acos(phi5_RHS) - beta, 2*math.pi)]
                    
                # print('Solutions for phi5 are ' + str(phi5_soln_arr))
                
            # Running through the solutions for phi1 and phi5, and checking if the final
            # configuration is reached            
            for phi1 in phi1_soln_arr:
                
                for phi5 in phi5_soln_arr:
                    
                    # Obtaining the final configuration of the path
                    _, _, _, fin_config_path, _, _, _ =\
                        points_path(np.identity(3), rL_mod, rR_mod, 1,\
                                    [phi1, phi2, phi3, phi4, phi3, phi5], path_type)
                        
                    # Checking if the minimum and maximum value in the difference in the final
                    # configurations is small                    
                    if abs(max(map(max, fin_config_path - fin_config_mod))) <= 10**(-6)\
                        and abs(min(map(min, fin_config_path - fin_config_mod))) <= 10**(-6):
                        
                        # Appending the path length, path parameters, and path cost
                        path_cost = R*(phi1 + 2*phi3) + rL*(1 + muL*Ul)*(phi2 + phi5) + rR*(1 + muR*Ur)*phi4
                        path_length = R*(phi1 + 2*phi3) + rL*(phi2 + phi5) + rR*phi4
                            
                        path_params.append([path_length, phi1, phi2, phi3, phi4, phi3, phi5, path_cost])
                        
    elif path_type == 'lgrglg':
        
        # Constructing the equation to solve for the middle G segment and the L and R segments
        A = -16*sqrt(1 - rL_mod**2)*Ul**2*Ur**2*mu
        B = 4*sqrt(1 - rL_mod**2)*Ur*mu*(-6*Ul*Ur*mu + 2*(1 - rL_mod**2)*Ur*mu**2\
                                         + Ul**2*(-4*mu + Ur*(11 + (1 - 2*rL_mod**2)*mu**2)))
        C = 2*mu**2*rL_mod**3*(Ul**2 + 1)*(2*mu**2*(rR_mod**2 - 1) + Ur**2*(mu**2*(2*rR_mod**2 - 1) - 1) + 2*mu*Ur)\
            - sqrt(1 - rL_mod**2)*(Ul**2*(4*mu**3 + Ur**2*(alpha13*(mu**2 - 1)**2 + 4*mu*(mu**2 + 10))\
                                          - 2*mu*Ur*(alpha13*(1 - mu**2) + mu**3 + 15*mu)) \
                                   + 2*mu*Ul*Ur*(2*mu*(alpha13 + 4*mu) + alpha13*(mu**2 - 1)*Ur\
                                                 - mu*(mu**2 + 21)*Ur) + 4*mu**3*Ur*(4*Ur - mu))\
            + 4*mu**3*sqrt(1 - rL_mod**2)*rL_mod**2*(Ul**2 + 1)*Ur*(2*Ur - mu)\
            - rL_mod*(Ul**2*(2*mu**2*(mu**2 + 1)*(rR_mod**2 - 1)\
                             + Ur**2*(alpha33*(mu**2 - 1)**2 + (mu**2 + 1)*(mu**2*(2*rR_mod**2 - 1) - 1)) \
                             + 2*Ur*mu*((alpha33 + 1)*mu**2 + 1 - alpha33))\
                      + 2*mu*Ul*(-2*mu**2*(rR_mod**2 - 1) + 2*(alpha33 - 1)*mu*Ur\
                                 + Ur**2*(alpha33*(mu**2 - 1) + mu**2*(1 - 2*rR_mod**2) + 1))\
                      + 2*mu**2*(2*mu**2*(rR_mod**2 - 1) + Ur**2*(mu**2*(2*rR_mod**2 - 1) - 1) + 2*mu*Ur))
        D = 2*mu**2*rL_mod**3*(Ul**2 + 1)*(Ur - mu)**2\
            + sqrt(1 - rL_mod**2)*(Ul**2*(mu**2*(alpha13*(1 - mu**2) + 4*mu) \
                                          + 2*Ur**2*(alpha13*(1 - mu**2) + 6*mu)\
                                          + 2*mu*Ur*(alpha13*(mu**2 - 2) - 7*mu)) \
                                   - 2*mu*Ul*(mu**2*(alpha13 + mu) + Ur**2*(alpha13*(2 - mu**2) + 9*mu)\
                                              - 4*mu*Ur*(alpha13 + 2*mu))\
                                   + mu**2*Ur*(Ur*(alpha13*(1 - mu**2) + 8*mu) - 2*mu*(alpha13 + 3*mu)))\
            - rL_mod*(Ul**2*(mu**2*(alpha33*(mu**2 - 1) + mu**2 - 2*rR_mod**2 + 3)\
                             + 2*Ur**2*(alpha33*(mu**2 - 1) - mu**2*(rR_mod**2 - 1) + 1) \
                             - 2*mu*Ur*(alpha33*(mu**2 - 2) + mu**2 + 2))\
                      - 2*mu*Ul*(-mu**2*(alpha33 + 2*rR_mod**2 - 3)\
                                 + Ur**2*(alpha33*(mu**2 - 2) + mu**2*(1 - 2*rR_mod**2) + 2) + 4*(alpha33 - 1)*mu*Ur)\
                      + mu**2*(-2*mu**2*(rR_mod**2 - 2) + 2*(alpha33 - 3)*mu*Ur\
                               + Ur**2*(alpha33*(mu**2 - 1) + mu**2*(1 - 2*rR_mod**2) + 3)))
        E = -(sqrt(1 - rL_mod**2)*alpha13 + rL_mod*(-1 + alpha33))*(Ul - mu)**2*(Ur - mu)**2
        
        coeff = [A, B, C, D, E]
        
        # # Obtaining the roots of the cubic polynomial
        x = np.roots(coeff)
        x_real = x.real[abs(x.imag) < 1e-5]
        print(x_real)
        
        phi2_soln_arr = []
        
        # Running through each solution in x
        for soln in x_real:
            
            # Checking if the value is in (0, 1)
            if soln > 0 and soln < 1:
                
                phi2_soln_arr.append(2*math.acos(math.sqrt(soln)))
                
        # print(phi2_soln_arr)
        
        # Running through each solution for phi2
        for phi2 in phi2_soln_arr:
            
            # Computing the solution for phi3 and phi4
            phi3 = math.pi + 2*math.atan2((Ur - mu)*math.tan(phi2/2), mu*math.sqrt(1 + Ur**2))
            phi4 = math.pi + 2*math.atan2((Ul - mu)*math.tan(phi2/2), mu*math.sqrt(1 + Ul**2))
            # print('Solution for phi3 is ' + str(phi3))
            # print('Solution for phi4 is ' + str(phi4))
            
            # Obtaining the solution for phi1
            beta11 = (cos(phi2))**2*(1 - rR_mod**2 + rR_mod**2*cos(phi3)) - cos(phi3)*(sin(phi2))**2 - rR_mod*sin(2*phi2)*sin(phi3)
            beta12 = -cos(phi2)*(1 - rR_mod**2 + (1 + rR_mod**2)*cos(phi3))*sin(phi2) - rR_mod*sin(phi3)*cos(2*phi2)
            beta13 = sqrt(1 - rR_mod**2)*(rR_mod*cos(phi2)*(-1 + cos(phi3)) - sin(phi2)*sin(phi3))
            beta22 = (cos(phi2))**2*cos(phi3) + (-1 + rR_mod**2 - rR_mod**2*cos(phi3))*(sin(phi2))**2 - rR_mod*sin(2*phi2)*sin(phi3)
            beta23 = sqrt(1 - rR_mod**2)*(rR_mod*sin(phi2)*(-1 + cos(phi3)) + cos(phi2)*sin(phi3))
            beta33 = rR_mod**2 + (1 - rR_mod**2)*cos(phi3)
            
            F = - rL_mod*(1 - rL_mod**2)*(rL_mod*(1 - cos(phi4))*beta11 - sin(phi4)*beta12)\
                - rL_mod*sqrt(1 - rL_mod**2)*(cos(phi4) + (2*rL_mod**2 - 1)*(1 - cos(phi4)))*beta13\
                + (1 - rL_mod**2)**(1.5)*sin(phi4)*beta23 + (1 - rL_mod**2)*(cos(phi4) + (1 - cos(phi4))*rL_mod**2)*beta33
            G = - (1 - rL_mod**2)*((1 - cos(phi4))*rL_mod*beta12 + sin(phi4)*beta22)\
                + sqrt(1 - rL_mod**2)*(cos(phi4) + rL_mod**2*(1 - cos(phi4)))*beta23
            H = rL_mod**2*(1 - rL_mod**2)*(1 - cos(phi4))*beta11 - rL_mod*(1 - rL_mod**2)*sin(phi4)*beta12\
                + rL_mod*sqrt(1 - rL_mod**2)*(cos(phi4) + 2*rL_mod**2*(1 - cos(phi4)))*beta13\
                + rL_mod**2*sqrt(1 - rL_mod**2)*sin(phi4)*beta23 + rL_mod**2*(cos(phi4) + (1 - cos(phi4))*rL_mod**2)*beta33
                
            if sqrt(F**2 + G**2) <= 10**(-6):
                        
                # print('phi1 cannot be solved for.')
                continue # Since phi1 cannot be solved for using the equation obtained
                
            else:
                
                gamma = math.atan2(G, F)
                
                phi1_RHS = (alpha33 - H)/sqrt(F**2 + G**2)
                
                # Checking if phi1_RHS is in [-1, 1] with a tolerance
                if abs(phi1_RHS) > 1 and abs(phi1_RHS) <= 1 + 10**(-6):
                    
                    phi1_RHS = np.sign(phi1_RHS)
                    
                elif abs(phi1_RHS) > 1:
                    
                    continue # Since solution for phi1 does not exist
                
                # Obtaining the solutions for phi1
                phi1_soln_arr = [np.mod(math.acos(phi1_RHS) + gamma, 2*math.pi),\
                                 np.mod(2*math.pi - math.acos(phi1_RHS) + gamma, 2*math.pi)]
                    
                # Checking if either solution is nearly equal to zero or 2*math.pi
                if 2*math.pi - phi1_soln_arr[0] <= 10**(-6) or phi1_soln_arr[0] <= 10**(-6):
                    
                    phi1_soln_arr[0] = 0
                    
                if 2*math.pi - phi1_soln_arr[1] <= 10**(-6) or phi1_soln_arr[1] <= 10**(-6):
                    
                    phi1_soln_arr[1] = 0
                    
                # Checking if the two solutions are equal
                if abs(phi1_soln_arr[1] - phi1_soln_arr[0]) <= 10**(-6):
                    
                    phi1_soln_arr = [np.mod(math.acos(phi1_RHS) + gamma, 2*math.pi)]
                    
                # print('Solutions for phi1 are ' + str(phi1_soln_arr))
                
            # Obtaining the solution for phi5
            I = sqrt(1 - rL_mod**2)*((1 - (1 - cos(phi3))*rR_mod**2)*(cos(phi2))**2\
                                     - rR_mod*sin(2*phi2)*sin(phi3) - (sin(phi2))**2*cos(phi3))\
                - rL_mod*sqrt(1 - rR_mod**2)*(rR_mod*(1 - cos(phi3))*cos(phi2) + sin(phi2)*sin(phi3))
            J = -sqrt(1 - rL_mod**2)*((1 - rR_mod**2 + (1 + rR_mod**2)*cos(phi3))*sin(phi2)*cos(phi2)\
                                      + rR_mod*cos(2*phi2)*sin(phi3))\
                - rL_mod*sqrt(1 - rR_mod**2)*(cos(phi2)*sin(phi3) - rR_mod*sin(phi2)*(1 - cos(phi3)))
            K = - sqrt(1 - rL_mod**2)*sqrt(1 - rR_mod**2)*(rR_mod*(1 - cos(phi3))*cos(phi2) + sin(phi2)*sin(phi3))\
                + rL_mod*(cos(phi3) + (1 - cos(phi3))*rR_mod**2)
            L = (1 - (1 - cos(phi4))*rL_mod**2)*I + rL_mod*sin(phi4)*J + (1 - cos(phi4))*rL_mod*sqrt(1 - rL_mod**2)*K
            M = -rL_mod*sin(phi4)*I + cos(phi4)*J + sin(phi4)*sqrt(1 - rL_mod**2)*K
            N = (1 - cos(phi4))*rL_mod*sqrt(1 - rL_mod**2)*I \
                - sin(phi4)*sqrt(1 - rL_mod**2)*J + (cos(phi4) + (1 - cos(phi4))*rL_mod**2)*K
                
            if math.sqrt(L**2 + M**2) <= 10**(-6):
                    
                continue # Since phi1 cannot be solved for using the equation obtained
                    
            else:
                
                beta = math.atan2(M, L)
                
                phi5_RHS = (alpha11 + rL_mod*math.sqrt(1 - rL_mod**2)*(alpha13 + alpha31)\
                            + rL_mod**2*(alpha33 - alpha11) - rL_mod*N)/(math.sqrt(1 - rL_mod**2)*math.sqrt(L**2 + M**2))
                
                # Checking if phi5_RHS is in [-1, 1] with a tolerance
                if abs(phi5_RHS) > 1 and abs(phi5_RHS) <= 1 + 10**(-6):
                    
                    phi5_RHS = np.sign(phi5_RHS)
                    
                if abs(phi5_RHS) > 1:
                    
                    continue # Since solution for phi5 does not exist
                
                # Obtaining the solutions for phi5
                phi5_soln_arr = [np.mod(math.acos(phi5_RHS) + beta, 2*math.pi),\
                                 np.mod(2*math.pi - math.acos(phi5_RHS) + beta, 2*math.pi)]
                    
                # Checking if either solution is nearly equal to zero or 2*math.pi
                if 2*math.pi - phi5_soln_arr[0] <= 10**(-6) or phi5_soln_arr[0] <= 10**(-6):
                    
                    phi5_soln_arr[0] = 0
                    
                if 2*math.pi - phi5_soln_arr[1] <= 10**(-6) or phi5_soln_arr[1] <= 10**(-6):
                    
                    phi5_soln_arr[1] = 0
                    
                # Checking if the two solutions are equal
                if abs(phi5_soln_arr[1] - phi5_soln_arr[0]) <= 10**(-6):
                    
                    phi5_soln_arr = [np.mod(math.acos(phi5_RHS) + beta, 2*math.pi)]
                    
                # print('Solutions for phi5 are ' + str(phi5_soln_arr))
                
            # Running through the solutions for phi1 and phi5, and checking if the final
            # configuration is reached            
            for phi1 in phi1_soln_arr:
                
                for phi5 in phi5_soln_arr:
                    
                    # Obtaining the final configuration of the path
                    _, _, _, fin_config_path, _, _, _ =\
                        points_path(np.identity(3), rL_mod, rR_mod, 1,\
                                    [phi1, phi2, phi3, phi2, phi4, phi5], path_type)
                        
                    # Checking if the minimum and maximum value in the difference in the final
                    # configurations is small                    
                    if abs(max(map(max, fin_config_path - fin_config_mod))) <= 10**(-6)\
                        and abs(min(map(min, fin_config_path - fin_config_mod))) <= 10**(-6):
                        
                        # Appending the path length, path parameters, and path cost
                        path_cost = R*(phi5 + 2*phi2) + rL*(1 + muL*Ul)*(phi1 + phi4) + rR*(1 + muR*Ur)*phi3
                        path_length = R*(phi5 + 2*phi2) + rL*(phi1 + phi4) + rR*phi3
                            
                        path_params.append([path_length, phi1, phi2, phi3, phi2, phi4, phi5, path_cost])
    
    # Checking if no solution was obtained for the path
    if len(path_params) == 0:
        
        print(path_type.upper() + ' path does not exist.')
        path_params.append([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
                
    return path_params

def paths_sphere(ini_config, fin_config, rL, rR, R, muL, muR, visualization = 1, filename = 'paths_sphere.html'):
    
    # Path types
    path_types_three_seg = np.array(['lgl', 'rgr', 'lgr', 'rgl', 'lrl'])
    
    # Plotting the initial and final configurations
    
    if visualization == 1:
        
        fig_3D = plotting_functions()
        # Plotting the sphere
        fig_3D.surface_3D(generate_points_sphere(np.array([0, 0, 0]), 1)[0],\
                          generate_points_sphere(np.array([0, 0, 0]), 1)[1],\
                          generate_points_sphere(np.array([0, 0, 0]), 1)[2], 'grey',\
                          'Sphere', 0.7)
        # Plotting the initial and final configurations
        fig_3D.points_3D([ini_config[0, 0]], [ini_config[1, 0]], [ini_config[2, 0]], 'red',\
                         'Initial point', 'circle')
        fig_3D.points_3D([fin_config[0, 0]], [fin_config[1, 0]],\
                         [fin_config[2, 0]], 'black', 'Final point', 'diamond')
        # Adding initial and final tangent vectors
        fig_3D.arrows_3D([ini_config[0, 0]], [ini_config[1, 0]], [ini_config[2, 0]],\
                          [ini_config[0, 1]], [ini_config[1, 1]], [ini_config[2, 1]],\
                          'orange', 'oranges', 'Initial tangent vector', 3, 1, 0.5, 'n')
        fig_3D.arrows_3D([fin_config[0, 0]], [fin_config[1, 0]],\
                         [fin_config[2, 0]], [fin_config[0, 1]],\
                         [fin_config[1, 1]], [fin_config[2, 1]],\
                          'green', 'greens', 'Final tangent vector', 3, 1, 0.5, 'n')
    
        fig_3D.update_layout_3D('X (m)', 'Y (m)', 'Z (m)', 'Initial and final configurations')
        # Writing the figure on the html file
        fig_3D.writing_fig_to_html(filename, 'w')
    
    # Generating the paths
    for path in path_types_three_seg:
        
        path_params = path_generation_sphere_three_seg(ini_config, fin_config, rL, rR, R, muL, muR, path)
        # Generating points along the path
        for possible_path in path_params:
            
            if ~np.isnan(possible_path[0]): # Checking if path exists
            
                x_coords_path, y_coords_path, z_coords_path, _, _, _, _ =\
                    points_path(ini_config, rL, rR, R, [possible_path[1], possible_path[2],\
                                                        possible_path[3]], path)
                # Plotting the path
                fig_3D_copy = copy.deepcopy(fig_3D)
                fig_3D_copy.scatter_3D(x_coords_path, y_coords_path, z_coords_path, 'blue', False)
                
                fig_3D_copy.update_layout_3D('X (m)', 'Y (m)', 'Z (m)', path.upper() + ' path')
                # Writing the figure on the html file
                fig_3D_copy.writing_fig_to_html(filename, 'a')
        
def optimal_path_sphere_three_seg(ini_config, fin_config, r, R,\
                                  visualization = 1, filename = 'paths_sphere.html'):
    
    # Path types
    path_types_three_seg = np.array(['lgl', 'rgr', 'lgr', 'rgl', 'lrl'])
    
    least_cost_path = 'lgl'
    least_cost_path_length = np.infty
    least_cost_path_params = []
    
    # Generating the paths
    for path in path_types_three_seg:
        
        path_params = path_generation_sphere_three_seg(ini_config, fin_config, r, r, R, 0, 0, path)
        
        # Checking if the considered path exists
        for possible_path in path_params:
            
            if ~np.isnan(possible_path[0]): # Checking if path exists
            
                # Updating the minimum cost path
                if possible_path[0] < least_cost_path_length:
                    
                    least_cost_path = path
                    least_cost_path_length = possible_path[0]
                    least_cost_path_params = possible_path[1:-1]
                    
    # Plotting the optimal path
    if visualization == 1:
        
        fig_3D = plotting_functions()
        # Plotting the sphere
        fig_3D.surface_3D(generate_points_sphere(np.array([0, 0, 0]), 1)[0],\
                          generate_points_sphere(np.array([0, 0, 0]), 1)[1],\
                          generate_points_sphere(np.array([0, 0, 0]), 1)[2], 'grey',\
                          'Sphere', 0.7)
        # Plotting the initial and final configurations
        fig_3D.points_3D([ini_config[0, 0]], [ini_config[1, 0]], [ini_config[2, 0]], 'red',\
                         'Initial point', 'circle')
        fig_3D.points_3D([fin_config[0, 0]], [fin_config[1, 0]],\
                         [fin_config[2, 0]], 'black', 'Final point', 'diamond')
        # Adding initial and final tangent vectors
        fig_3D.arrows_3D([ini_config[0, 0]], [ini_config[1, 0]], [ini_config[2, 0]],\
                          [ini_config[0, 1]], [ini_config[1, 1]], [ini_config[2, 1]],\
                          'orange', 'oranges', 'Initial tangent vector', 3, 1, 0.5, 'n')
        fig_3D.arrows_3D([fin_config[0, 0]], [fin_config[1, 0]],\
                         [fin_config[2, 0]], [fin_config[0, 1]],\
                         [fin_config[1, 1]], [fin_config[2, 1]],\
                          'green', 'greens', 'Final tangent vector', 3, 1, 0.5, 'n')
    
        fig_3D.update_layout_3D('X (m)', 'Y (m)', 'Z (m)', 'Initial and final configurations')
        # Writing the figure on the html file
        fig_3D.writing_fig_to_html(filename, 'w')               
        
        # Plotting the optimal path
        x_coords_path, y_coords_path, z_coords_path, _, _, _, _ =\
            points_path(ini_config, r, r, R, least_cost_path_params, least_cost_path)
        # Plotting the path
        fig_3D_copy = copy.deepcopy(fig_3D)
        fig_3D_copy.scatter_3D(x_coords_path, y_coords_path, z_coords_path, 'blue', False)
        
        fig_3D_copy.update_layout_3D('X (m)', 'Y (m)', 'Z (m)', least_cost_path.upper() + ' path')
        # Writing the figure on the html file
        fig_3D_copy.writing_fig_to_html(filename, 'a')
    
    return least_cost_path, least_cost_path_length, least_cost_path_params

def generate_random_configs(R):
    '''
    This function generates random initial and final configurations on a sphere
    of radius R centered at the origin.

    Parameters
    ----------
    R : Scalar
        Radius of the sphere.

    Returns
    -------
    ini_config : Numpy array
        Contains the initial configuration. The syntax followed is as
        follows:
            The first column contains the position.
            The second column contains the tangent vector.
            The third column contains the tangent-normal vector.
    fin_config : Numpy array
        Contains the final configuration. The same syntax used for ini_config is
        used here.

    '''
    
    # Generate random longitude and colatitude for the sphere for the initial position
    phi_ini = np.random.rand()*math.pi
    theta_ini = np.random.rand()*2*math.pi
    xini = np.array([R*sin(phi_ini)*cos(theta_ini), R*sin(phi_ini)*sin(theta_ini), R*cos(phi_ini)])
    # Generating the final location
    phi_fin = np.random.rand()*math.pi
    theta_fin = np.random.rand()*2*math.pi
    xfin = np.array([R*sin(phi_fin)*cos(theta_fin), R*sin(phi_fin)*sin(theta_fin), R*cos(phi_fin)])
    
    # Generating random initial and final heading vectors and orthonormalizing them
    # with respect to the respect location's outward facing surface normal
    rand_ini_vect = np.random.rand(3,)
    rand_fin_vect = np.random.rand(3,)
    
    # Orthonormalizing rand_ini_vect with respect to a unit vector along xini
    t_ini = rand_ini_vect - np.dot(rand_ini_vect, xini/R)*xini/R
    if np.linalg.norm(t_ini) <= 10**(-6):
        
        raise Exception('Regenerate the initial tangent vector.')
        
    else:
        
        t_ini = t_ini/np.linalg.norm(t_ini)
    
    # Orthonormalizing rand_fin_vect with respect to a unit vector along xfin
    t_fin = rand_fin_vect - np.dot(rand_fin_vect, xfin/R)*xfin/R
    if np.linalg.norm(t_fin) <= 10**(-6):
        
        raise Exception('Regenerate the final tangent vector.')
        
    else:
        
        t_fin = t_fin/np.linalg.norm(t_fin)
        
    # Constructing the initial configuration matrix
    # Obtaining the tangent-normal vector
    T_ini = np.cross(xini/R, t_ini)
    ini_config = np.array([[xini[0], t_ini[0], T_ini[0]],\
                           [xini[1], t_ini[1], T_ini[1]],\
                           [xini[2], t_ini[2], T_ini[2]]])
    
    # Constructing the final configuration matrix
    # Obtaining the tangent-normal vector
    T_fin = np.cross(xfin/R, t_fin)
    fin_config = np.array([[xfin[0], t_fin[0], T_fin[0]],\
                           [xfin[1], t_fin[1], T_fin[1]],\
                           [xfin[2], t_fin[2], T_fin[2]]])
        
    return ini_config, fin_config    
        
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