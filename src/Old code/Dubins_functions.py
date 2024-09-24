# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:36:57 2021

@author: deepa
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.graph_objs import *
init_notebook_mode()

def tight_circles(loc_head_vec, rad_turn, visualization = 1):
    
    '''
    DESCRIPTION: This function takes in the location and the radius of the turn\
    as inputs and returns the left and right tight circles.
    INPUTS:
    loc_head_vec is a numpy array with length = 3. The first two indeces\
    contain the x and y coordinates in meters. The third coordinate contains the\
    heading angle in radians.
    rad_turn is the radius of the tight circle turn in meters.
    visualization = 1 if the plot should be generated for visualizing the left\
    and right turns, and is 0 if plot should not be generated. 
    OUTPUTS:
    Coordinates of the left and right tight circles are returned as a numpy matrix,\
    where the first row corresponds to the first circle's center and the second\
    row corresponds to the second circle's center.
    '''
    
    x_loc = loc_head_vec[0];
    y_loc = loc_head_vec[1];
    heading_ang = loc_head_vec[2];
    
    # Left tight circle turn's center
    x_l = x_loc + rad_turn*math.cos(heading_ang + math.pi/2)
    y_l = y_loc + rad_turn*math.sin(heading_ang + math.pi/2)
    
    # Right tight circle turn's center
    x_r = x_loc + rad_turn*math.cos(heading_ang - math.pi/2)
    y_r = y_loc + rad_turn*math.sin(heading_ang - math.pi/2)
    
    # if visualization == 1:
    
    #     plt.arrow(x_loc, y_loc, math.cos(heading_ang), math.sin(heading_ang),\
    #               length_includes_head = True, head_width = .1)
    #     circle1 = plt.Circle((x_l, y_l), rad_turn, color = 'r', fill = False)
    #     plt.gca().add_patch(circle1)
    #     circle2 = plt.Circle((x_r, y_r), rad_turn, color = 'g', fill = False)
    #     plt.gca().add_patch(circle2)
    #     plt.show()
        
    if visualization == 1:
        
        fig = go.Figure()
        
        plots = []
        
        # Location and heading
        # arrow = go.Scatter3d(
        #     x = [xloc, xloc + 0.75*rad_turn*math.cos(heading_ang)],
        #     y = [yloc, yloc + 0.75*rad_turn*math.sin(heading_ang)],
        #     marker = dict(
        #     size = 0.2,
        #     color = 'black',
        #     colorscale = 'Viridis',),
        #     line = dict(
        #     color = 'black',
        #     width = 5),
        #     name = 'Configuration')
        # arrowtipsize = 5
        # arrowtip = go.Cone(
        #         x = [xloc + 0.75*rad_turn*math.cos(heading_ang)],
        #         y = [yloc + 0.75*rad_turn*math.cos(heading_ang)],
        #         u=[arrowtipsize*Ad[2][i][0]],
        #         v=[arrowtipsize*Ad[2][i][1]],
        #         w=[arrowtipsize*Ad[2][i][2]],
        #         colorscale = Ad[3][i],
        #         showscale = False,
        #         name = tracename)
        fig = ff.create_quiver([x_loc], [y_loc], [math.cos(heading_ang)],\
                                 [math.sin(heading_ang)], scale=.25,\
                                 arrow_scale=.4, name = 'Configuration', line_width = 1)
        # Adding the tight circles
        theta = np.linspace(0, 2*math.pi, 100)
        circle_1_pts_x_coord = np.array([x_l + rad_turn*math.cos(i) for i in theta])
        circle_1_pts_y_coord = np.array([y_l + rad_turn*math.sin(i) for i in theta])
        circle_2_pts_x_coord = np.array([x_r + rad_turn*math.cos(i) for i in theta])
        circle_2_pts_y_coord = np.array([y_r + rad_turn*math.sin(i) for i in theta])
        
        fig.add_trace(go.Scatter(x = circle_1_pts_x_coord, y = circle_1_pts_y_coord,\
                                 marker=dict(color= 'green', colorscale='Viridis'),
                                 line = dict(width = 2), name = 'Left circle'))
        fig.add_trace(go.Scatter(x = circle_2_pts_x_coord, y = circle_2_pts_y_coord,\
                                 marker=dict(color= 'red', colorscale='Viridis'),
                                 line = dict(width = 2), name = 'Right circle'))
        
        fig.update_layout(title = 'Configuration with left and right tight circles')
            
        plot(fig)
    
    return np.array([[x_l, y_l], [x_r, y_r]])

def tangent_pts(circle_1_center, circle_2_center, ini_config, fin_config, rad_turn,\
                tang_type = 'i', visualization = 1):
    
    '''
    DESCRIPTION:
    This function takes in the location of centers of two circles\
    and the radius of the two circles, and determines the tangent points for\
    the two inner tangents or the two outer tangents depending on the argument.
    INPUTS:
    circle_1_center and circle_2_center are numpy arrays of length 2, and contain\
    the x and y coordinates of circle 1 and circle 2, respectively.
    ini_config and fin_config are 1x3 numpy arrays containing the initial and\
    final configurations.
    rad_turn is the radius of the two circles.
    tang_type is an argument to fix whether the inner tangent points or the outer\
    tangent points should be calculated. 'i' => inner tangent, 'o' => outer tangent.
    visualization = 1 if the plot should be generated for visualizing the left\
    and right turns, and is 0 if plot should not be generated.
    OUTPUTS:
    The tangent points on each circle corresponding to the two inner tangents\
    or the two outer tangents are obtained. The outputs are provided as a numpy\
    matrix, where the first two rows are the tangent points of tangent 1 and 2,\
    respectively, on the first circle, and the last two rows are the tangent\
    points of tangent 1 and 2, respectively, on the second circle.
    The length of the corresponding tangent is returned under length_tangent.
    '''
    
    # Calculating the length of the vector connecting the two circles' centers.
    vect_v = np.array([circle_2_center[0] - circle_1_center[0], circle_2_center[1] - circle_1_center[1]])
    length_v = np.linalg.norm(vect_v)
    
    # Calculating the angle of the vector connecting the two circles' centers.
    phi = math.atan2(vect_v[1], vect_v[0])
    
    if tang_type == 'i':
        
        # Dot product between a unit vector along the vector joining the two
        # circles' centers and the unit vector perpendicular to an inner tangent
        dot_prod_vect_v_tang = 2*rad_turn/length_v
        
        # Checking for existance of inner tangents
        if dot_prod_vect_v_tang > 1: # In this case, inner tangent does not exist
            
            x1t3 = np.NaN
            y1t3 = np.NaN
            x1t4 = np.NaN
            y1t4 = np.NaN
            x2t3 = np.NaN
            y2t3 = np.NaN
            x2t4 = np.NaN
            y2t4 = np.NaN
            length_tangent = np.NaN
            
        else: # In this case, the inner tangent exists
        
            # Tangent points on first circle - first tangent
            x1t3 = circle_1_center[0] + math.cos(phi)*rad_turn*dot_prod_vect_v_tang \
                - rad_turn*math.sin(phi)*math.sqrt(1 - dot_prod_vect_v_tang**2)
            y1t3 = circle_1_center[1] + math.sin(phi)*rad_turn*dot_prod_vect_v_tang \
                + rad_turn*math.cos(phi)*math.sqrt(1 - dot_prod_vect_v_tang**2)
            # Tangent points on first circle - second tangent
            x1t4 = circle_1_center[0] + math.cos(phi)*rad_turn*dot_prod_vect_v_tang \
                + rad_turn*math.sin(phi)*math.sqrt(1 - dot_prod_vect_v_tang**2)
            y1t4 = circle_1_center[1] + math.sin(phi)*rad_turn*dot_prod_vect_v_tang \
                - rad_turn*math.cos(phi)*math.sqrt(1 - dot_prod_vect_v_tang**2)
                
            # Tangent points on second circle - first tangent
            x2t3 = circle_2_center[0] - math.cos(phi)*rad_turn*dot_prod_vect_v_tang \
                + rad_turn*math.sin(phi)*math.sqrt(1 - dot_prod_vect_v_tang**2)
            y2t3 = circle_2_center[1] - math.sin(phi)*rad_turn*dot_prod_vect_v_tang \
                - rad_turn*math.cos(phi)*math.sqrt(1 - dot_prod_vect_v_tang**2)
            # Tangent points on second circle - second tangent
            x2t4 = circle_2_center[0] - math.cos(phi)*rad_turn*dot_prod_vect_v_tang \
                - rad_turn*math.sin(phi)*math.sqrt(1 - dot_prod_vect_v_tang**2)
            y2t4 = circle_2_center[1] - math.sin(phi)*rad_turn*dot_prod_vect_v_tang \
                + rad_turn*math.cos(phi)*math.sqrt(1 - dot_prod_vect_v_tang**2)
            
            # Length of the tangent
            length_tangent = math.sqrt(length_v**2 - 4*rad_turn**2)
            
            if visualization == 1:
        
                circle1 = plt.Circle((circle_1_center[0], circle_1_center[1]), rad_turn,\
                                     color = 'r', fill = False)
                plt.gca().add_patch(circle1)
                circle2 = plt.Circle((circle_2_center[0], circle_2_center[1]), rad_turn,\
                                     color = 'g', fill = False)
                plt.gca().add_patch(circle2)
                plt.arrow(ini_config[0], ini_config[1], math.cos(ini_config[2]), math.sin(ini_config[2]),\
                      length_includes_head = True, head_width = .1)
                plt.arrow(fin_config[0], fin_config[1], math.cos(fin_config[2]), math.sin(fin_config[2]),\
                      length_includes_head = True, head_width = .1)
                plt.plot([x1t3, x2t3], [y1t3, y2t3])
                plt.plot([x1t4, x2t4], [y1t4, y2t4])
                plt.scatter([circle_1_center[0], circle_2_center[0]],\
                            [circle_1_center[1], circle_2_center[1]], marker = '*')
                plt.show()
        
        return np.array([[x1t3, y1t3], [x1t4, y1t4], [x2t3, y2t3], [x2t4, y2t4]]), length_tangent
        
    elif tang_type == 'o':
        
        # Tangent points on first circle - first tangent
        x1t1 = circle_1_center[0] + rad_turn*math.cos(phi + math.pi/2)
        y1t1 = circle_1_center[1] + rad_turn*math.sin(phi + math.pi/2)
        # Tangent points on first circle - second tangent
        x1t2 = circle_1_center[0] + rad_turn*math.cos(phi - math.pi/2)
        y1t2 = circle_1_center[1] + rad_turn*math.sin(phi - math.pi/2)
        
        # Tangent points on second circle - first tangent
        x2t1 = circle_2_center[0] + rad_turn*math.cos(phi + math.pi/2)
        y2t1 = circle_2_center[1] + rad_turn*math.sin(phi + math.pi/2)
        # Tangent points on second circle - second tangent
        x2t2 = circle_2_center[0] + rad_turn*math.cos(phi - math.pi/2)
        y2t2 = circle_2_center[1] + rad_turn*math.sin(phi - math.pi/2)
        
        # Length of the tangent
        length_tangent = length_v
        
        if visualization == 1:
    
            circle1 = plt.Circle((circle_1_center[0], circle_1_center[1]), rad_turn,\
                                 color = 'r', fill = False)
            plt.gca().add_patch(circle1)
            circle2 = plt.Circle((circle_2_center[0], circle_2_center[1]), rad_turn,\
                                 color = 'g', fill = False)
            plt.gca().add_patch(circle2)
            plt.arrow(ini_config[0], ini_config[1], math.cos(ini_config[2]), math.sin(ini_config[2]),\
                  length_includes_head = True, head_width = .1)
            plt.arrow(fin_config[0], fin_config[1], math.cos(fin_config[2]), math.sin(fin_config[2]),\
                  length_includes_head = True, head_width = .1)
            plt.plot([x1t1, x2t1], [y1t1, y2t1])
            plt.plot([x1t2, x2t2], [y1t2, y2t2])
            plt.scatter([circle_1_center[0], circle_2_center[0]],\
                        [circle_1_center[1], circle_2_center[1]], marker = '*')
            plt.show()
        
        return np.array([[x1t1, y1t1], [x1t2, y1t2], [x2t1, y2t1], [x2t2, y2t2]]), length_tangent
        
    else:
        
        raise Exception('Incorrect tangent input type')
        
def angle_of_turn(ini_pt, fin_pt, circle_center, rad_turn, sense_turn = 'l'):
    '''
    DESCRIPTION:
    This function takes in the coordinates of the initial and final points, the\
    coordinates of the center of the circle and the radius of the circle on which\
    the points lie, and the sense of the turn to return the angle of the turn and\
    the distance of the path
    INPUTS:
    ini_pt, fin_pt, circle_center are numpy arrays of length 2, and contain\
    the X and Y coordinates of the initial point, final point, and the center of\
    the circle, respectively.
    NOTE: ini_pt and fin_pt can also be the initial and final configurations.
    rad_turn is the radius of the circle on which the two points lie.
    sense_turn = 'l' if the direction of the turn is left, and 'r' if the direction\
    of the turn is right.
    OUTPUTS:
    angle_turn and path_length are the angle of the turn from the initial to the\
    final point and the distance travelled on the circle, respectively.
    '''
    
    # Vector connecting the center of the circle to the two points
    vect_ini = np.array([ini_pt[0] - circle_center[0], ini_pt[1] - circle_center[1]])
    vect_fin = np.array([fin_pt[0] - circle_center[0], fin_pt[1] - circle_center[1]])
    
    # Phase of these two vectors
    phase_vect_ini = math.atan2(vect_ini[1], vect_ini[0])
    phase_vect_fin = math.atan2(vect_fin[1], vect_fin[0])
    
    if sense_turn == 'l' and (phase_vect_fin - phase_vect_ini) >= 0:
        
        angle_turn = phase_vect_fin - phase_vect_ini
        path_length = abs(rad_turn*angle_turn)
        
    elif sense_turn == 'l' and (phase_vect_fin - phase_vect_ini) < 0:
        
        angle_turn = phase_vect_fin - phase_vect_ini + 2*math.pi
        path_length = abs(rad_turn*angle_turn)
        
    elif sense_turn == 'r' and (phase_vect_fin - phase_vect_ini) <= 0:
        
        angle_turn = phase_vect_fin - phase_vect_ini
        path_length = abs(rad_turn*angle_turn)
        
    elif sense_turn == 'r' and (phase_vect_fin - phase_vect_ini) > 0:
        
        angle_turn = phase_vect_fin - phase_vect_ini - 2*math.pi
        path_length = abs(rad_turn*angle_turn)
        
    else:
        
        raise Exception('Incorrect direction of turn')
    
    return angle_turn, path_length

def choosing_appropriate_tangent(ini_config, circle_center, rad_turn, tang_coordinates,\
                                 sense_turn = 'l'):
    '''
    DESCRIPTION:
    This function takes in the coordinates of the initial point, the center of the circle\
    about which the turn is made, the sense of the turn, and the coordinates of the\
    two possible tangents. This function can be used for path types wherein first\
    a tight circle turn is made followed by a straight line path is taken.
    INPUTS:
    ini_config and circle_center are numpy arrays of length 2, and contain\
    the X and Y coordinates of the initial point and the center of\
    the circle, respectively.
    rad_turn is the radius of the turn of the circle.
    tang_coordinates contains the coordinates of the two possible tangents \
    (two inner or two outer tangents) as a numpy array, wherein the first two rows\
    are the start and end points of the first tangent, respectively, and the last\
    two rows are the start and end points, respectively, of the second tangent. These\
    can be obtained from the function tangent_pts.
    sense_turn = 'l' if the direction of the turn is left, and 'r' if the direction\
    of the turn is right.
    OUTPUTS:
    The coordinates of the start and end points of the appropriate tangent are\
    returned as a numpy array, wherein the first row contains the coordinates of\
    the start point of the tangent and the second row contains the coordinates of\
    the end point of the tangent.
    The angle of the turn to reach the tangent point corresponding to the appropriate\
    tangent is returned under angle_turn_reach_tangpt
    The length of the path taken to reach the tangent point corresponding to the\
    appropriate tangent is returned under path_length_on_circle
    '''
    
    # Angle of turn to reach tangent pt 1 and tangent pt 2
    angle_turn_pt1, path_length_1 = angle_of_turn(ini_config, tang_coordinates[0, :],\
                                                  circle_center, rad_turn, sense_turn)
    angle_turn_pt2, path_length_2 = angle_of_turn(ini_config, tang_coordinates[1, :],\
                                                  circle_center, rad_turn, sense_turn)
    
    # Unit vectors along the heading at the tangent points    
    heading_pt1_vec = np.array([math.cos(ini_config[2] + angle_turn_pt1),\
                               math.sin(ini_config[2] + angle_turn_pt1)])
    heading_pt2_vec = np.array([math.cos(ini_config[2] + angle_turn_pt2),\
                               math.sin(ini_config[2] + angle_turn_pt2)])
        
    # Vectors corresponding to the two tangents
    tang_1_vec = np.array([tang_coordinates[2, 0] - tang_coordinates[0, 0],\
                           tang_coordinates[2, 1] - tang_coordinates[0, 1]])
    tang_2_vec = np.array([tang_coordinates[3, 0] - tang_coordinates[1, 0],\
                           tang_coordinates[3, 1] - tang_coordinates[1, 1]])
    
    # Taking the dot product with the vectors corresponding to the two possible tangents
    if np.dot(tang_1_vec, heading_pt1_vec) > 0:
        
        # Choosing the first tangent point
        x_tang_ini = tang_coordinates[0, 0]
        y_tang_ini = tang_coordinates[0, 1]
        # Choosing the tangent points corresponding to the second circle
        x_tang_fin = tang_coordinates[2, 0]
        y_tang_fin = tang_coordinates[2, 1]
        
        # Assigning the angle of turn to reach the tangent point and the length of the path
        # taken to reach the tangent point
        angle_turn_reach_tangpt = angle_turn_pt1
        path_length_on_circle = path_length_1
    
    elif np.dot(tang_2_vec, heading_pt2_vec) > 0:
        
        # Choosing the second tangent point
        x_tang_ini = tang_coordinates[1, 0]
        y_tang_ini = tang_coordinates[1, 1]
        # Choosing the tangent points corresponding to the second circle
        x_tang_fin = tang_coordinates[3, 0]
        y_tang_fin = tang_coordinates[3, 1]
        
        # Assigning the angle of turn to reach the tangent point and the length of the path
        # taken to reach the tangent point
        angle_turn_reach_tangpt = angle_turn_pt2
        path_length_on_circle = path_length_2
    
    return np.array([[x_tang_ini, y_tang_ini], [x_tang_fin, y_tang_fin]]),\
        angle_turn_reach_tangpt, path_length_on_circle
        
# Path types

def CSC_path(ini_config, fin_config, rad_turn, path_type = 'LSL', visualization = 1):
    '''
    DESCRIPTION:
    This function generates an CSC path given the initial configuration, final\
    configuration, and the radius of the tight turn.

    Parameters
    ----------
    ini_config : NUMPY ARRAY (1x3)
        Contains the initial configuration, i.e., the initial position and the initial heading.
    fin_config : NUMPY ARRAY (1x3)
        Contains the final configuration, i.e., the final position and the initial heading.
    rad_turn : Scalar
        Radius of the tight turn in m.
    path_type : String
        'LSL', 'RSR', 'LSR', and 'RSL' are possible path types that can be generated using\
        this function.
    visualization : Scalar
        If equal to 1, the LSL path is plotted. If set to zero, plot is not generated.

    Returns
    -------
    path_length : Scalar
        The length of the path is returned.
    initial_circle, final_circle : Numpy 1x2 arrays
        Contains the centers of the initial and final tight circle turns.
    tang_pts_appropriate : Numpy 2x2 array
        Returns the tangent points on the initial and final tight circles connecting which
        the straight line path can be obtained.

    '''
    
    if path_type.lower()[1] != 's':
        
        raise Exception('Incorrect path type. Middle segment needs to be a straight line.')
    
    # Constructing the circles at the initial and final configurations
    circle = tight_circles(ini_config, rad_turn, 0)
    if path_type.lower()[0] == 'l':
        
        initial_circle = circle[0, :]
        
    elif path_type.lower()[0] == 'r':
        
        initial_circle = circle[1, :]
        
    else:
        
        raise Exception('Incorrect path type. First segment must be a left or right turn.')
        
    circle = tight_circles(fin_config, rad_turn, 0)
    if path_type.lower()[2] == 'l':
        
        final_circle = circle[0, :]
        
    elif path_type.lower()[2] == 'r':
        
        final_circle = circle[1, :]
        
    else:
        
        raise Exception('Incorrect path type. Third segment must be a left or right turn.')
    
    # Obtaining the tangent points corresponding to outer tangent connections for lsl and rsr path types,
    # and corresponding to inner tangent connections for lsr and rsl path types.
    if path_type.lower()[0] == path_type.lower()[2]: # lsl and rsr path types
    
        tang_pts, l_tangent = tangent_pts(initial_circle, final_circle, ini_config, fin_config,\
                                          rad_turn, 'o', 0)
            
    else: # lsr and rsl path types
        
        tang_pts, l_tangent = tangent_pts(initial_circle, final_circle, ini_config, fin_config,\
                                          rad_turn, 'i', 0)
            
    if np.isnan(l_tangent) == False: # Checking if the tangent exists, in particular, for inner tangent
    # In this case, tangent exists
        
        # Choosing the appropriate tangent from the two tangents depending on the sense of the first turn
        tang_pts_appropriate, angle_1, path_length_cir1 = choosing_appropriate_tangent(ini_config,\
                                                           initial_circle, rad_turn, tang_pts,\
                                                           path_type.lower()[0])
            
        # Angle to reach the final point from the tangent point on the second circle
        angle_2, path_length_cir2 = angle_of_turn(tang_pts_appropriate[1, :], fin_config,\
                                                  final_circle, rad_turn, path_type.lower()[2])
        
        # Total path length
        path_length = path_length_cir1 + l_tangent + path_length_cir2
                
        if visualization == 1:
            
            plt.figure()
            # Plotting the initial and final configurations
            plt.arrow(ini_config[0], ini_config[1], math.cos(ini_config[2]), math.sin(ini_config[2]),\
                  length_includes_head = True, head_width = .2, color = 'orange', label = 'Initial configuration')
            plt.arrow(fin_config[0], fin_config[1], math.cos(fin_config[2]), math.sin(fin_config[2]),\
                  length_includes_head = True, head_width = .2, color = 'black', label = 'Final configuration')
            # Plotting the initial and final circles
            circle1 = plt.Circle((initial_circle[0], initial_circle[1]), rad_turn,\
                                      color = 'r', fill = False, label = 'Initial circle')
            plt.gca().add_patch(circle1)
            circle2 = plt.Circle((final_circle[0], final_circle[1]), rad_turn,\
                                  color = 'g', fill = False, label = 'Final circle')
            plt.gca().add_patch(circle2)
            
            # Plotting the path
            # Path on the first circle
            theta_1 = np.linspace(ini_config[2], ini_config[2] + angle_1, 100)
            # NOTE: theta_1 provides discretization of the headings as the Dubins vehicle moves from the initial
            # location to the tangent point. The angle traced on the circle, which is on the left or right of the initial
            # location is equal to the instanteous heading minus or plus pi/2, respectivley. This is because the vector
            # connecting the center of the circle with the instaneous configuration has a phase angle equal to the
            # instaneous heading minus (for left turn) or plus (for right turn) pi/2.
            
            if path_type.lower()[0] == 'l':
            
                pts_circle_1_x_coord = np.array([initial_circle[0] + rad_turn*math.cos(i - math.pi/2) for i in theta_1])
                pts_circle_1_y_coord = np.array([initial_circle[1] + rad_turn*math.sin(i - math.pi/2) for i in theta_1])
                
            else:
                
                pts_circle_1_x_coord = np.array([initial_circle[0] + rad_turn*math.cos(i + math.pi/2) for i in theta_1])
                pts_circle_1_y_coord = np.array([initial_circle[1] + rad_turn*math.sin(i + math.pi/2) for i in theta_1])
                
            # Path on the second circle
            theta_2 = np.linspace(fin_config[2] - angle_2, fin_config[2], 100)
            # Same note as that for theta_1
            
            if path_type.lower()[2] == 'l':
                
                pts_circle_2_x_coord = np.array([final_circle[0] + rad_turn*math.cos(i - math.pi/2) for i in theta_2])
                pts_circle_2_y_coord = np.array([final_circle[1] + rad_turn*math.sin(i - math.pi/2) for i in theta_2])
            
            else:
                
                pts_circle_2_x_coord = np.array([final_circle[0] + rad_turn*math.cos(i + math.pi/2) for i in theta_2])
                pts_circle_2_y_coord = np.array([final_circle[1] + rad_turn*math.sin(i + math.pi/2) for i in theta_2])
            
            # x coordinates for the complete path
            x_coords_path = np.append(np.append(pts_circle_1_x_coord, np.array([tang_pts_appropriate[0, 0],\
                                                                            tang_pts_appropriate[1, 0]])),\
                                  pts_circle_2_x_coord)
            # y coordinates for the complete path
            y_coords_path = np.append(np.append(pts_circle_1_y_coord, np.array([tang_pts_appropriate[0, 1],\
                                                                            tang_pts_appropriate[1, 1]])),\
                                  pts_circle_2_y_coord)
            # Plotting the complete path
            plt.plot(x_coords_path, y_coords_path, color = 'blue', label = path_type.upper() + ' path')
            
            # Showing the centers of the two circles
            plt.scatter([initial_circle[0], final_circle[0]], [initial_circle[1], final_circle[1]], marker = '*')
            
            plt.legend()
            
            plt.show()
            
    else: # In this case, tangent does not exist
    
        path_length = tang_pts_appropriate = angle_1 = angle_2 = np.NaN
        print('The path', path_type.upper(), 'does not exist.')
        
    return path_length, initial_circle, final_circle, tang_pts_appropriate, angle_1, angle_2

def CCC_path(ini_config, fin_config, rad_turn, path_type = 'LRL', visualization = 1):
    '''
    DESCRIPTION:
    This function generates an CCC path given the initial configuration, final\
    configuration, and the radius of the tight turn.

    Parameters
    ----------
    ini_config : NUMPY ARRAY (1x3)
        Contains the initial configuration, i.e., the initial position and the initial heading.
    fin_config : NUMPY ARRAY (1x3)
        Contains the final configuration, i.e., the final position and the initial heading.
    rad_turn : Scalar
        Radius of the tight turn in m.
    path_type : String
        'LRL' and 'RLR' are possible path types that can be generated using\
        this function.
    visualization : Scalar
        If equal to 1, the LSL path is plotted. If set to zero, plot is not generated.

    Returns
    -------
    path_length : Scalar
        The length of the path is returned.

    '''
    
    if path_type.lower() != 'lrl' and path_type.lower() != 'rlr':
        
        raise Exception('Incorrect path type.')  
        
    # Constructing the circles at the initial and final configurations
    circle = tight_circles(ini_config, rad_turn, 0)
    if path_type.lower()[0] == 'l':
        
        initial_circle = circle[0, :]
        
    elif path_type.lower()[0] == 'r':
        
        initial_circle = circle[1, :]
        
    circle = tight_circles(fin_config, rad_turn, 0)
    if path_type.lower()[2] == 'l':
        
        final_circle = circle[0, :]
        
    elif path_type.lower()[2] == 'r':
        
        final_circle = circle[1, :]
    
    # Distance between the two circles
    vec_connecting_centers = final_circle - initial_circle
    
    # Checking existance of the path
    if np.linalg.norm(vec_connecting_centers) > 4*rad_turn:
    # In this case, path does not exist
        
        path_length = middle_circle_1 = middle_circle_2 =  tang_pt_11\
            = tang_pt_31 = tang_pt_12 = tang_pt_32 = np.array([np.NaN, np.NaN])
        angle_turn_11 = angle_turn_21 = angle_turn_31 = angle_turn_12\
            = angle_turn_22 = angle_turn_32 = np.NaN
        print('The path', path_type.upper(), 'does not exist.')
        
    else:
        
        # Phase angle of the vector connecting the centers of the two circles
        phase_vec = math.atan2(vec_connecting_centers[1], vec_connecting_centers[0])
        # Two paths exist for the given path
        # Phase angle for the tangent points relative to the vector connecting the
        # centers of the two circles
        phase_rel_tang_pts = math.acos(np.linalg.norm(vec_connecting_centers)/(4*rad_turn))
        
        # Path number 1 - tangent point on the first circle
        tang_pt_11 = np.array([initial_circle[0] + rad_turn*math.cos(phase_vec + phase_rel_tang_pts),\
                              initial_circle[1] + rad_turn*math.sin(phase_vec + phase_rel_tang_pts)])
        # Path number 1 - tangent point on the second circle
        tang_pt_31 = np.array([final_circle[0] + rad_turn*math.cos(math.pi + phase_vec - phase_rel_tang_pts),\
                              final_circle[1] + rad_turn*math.sin(math.pi + phase_vec - phase_rel_tang_pts)])
        # Path number 1 - center of the middle circle
        middle_circle_1 = np.array([initial_circle[0] + 2*rad_turn*math.cos(phase_vec + phase_rel_tang_pts),\
                              initial_circle[1] + 2*rad_turn*math.sin(phase_vec + phase_rel_tang_pts)])
        
        # Path number 2 - tangent point on the first circle
        tang_pt_12 = np.array([initial_circle[0] + rad_turn*math.cos(phase_vec - phase_rel_tang_pts),\
                              initial_circle[1] + rad_turn*math.sin(phase_vec - phase_rel_tang_pts)])
        # Path number 2 - tangent point on the second circle
        tang_pt_32 = np.array([final_circle[0] + rad_turn*math.cos(math.pi + phase_vec + phase_rel_tang_pts),\
                              final_circle[1] + rad_turn*math.sin(math.pi + phase_vec + phase_rel_tang_pts)])
        # Path number 2 - center of the middle circle
        middle_circle_2 = np.array([initial_circle[0] + 2*rad_turn*math.cos(phase_vec - phase_rel_tang_pts),\
                              initial_circle[1] + 2*rad_turn*math.sin(phase_vec - phase_rel_tang_pts)])
            
        # Computing angles of turn and the path length
        # Path number 1
        angle_turn_11, path_length_11 = angle_of_turn(ini_config, tang_pt_11, initial_circle, rad_turn, path_type.lower()[0])
        angle_turn_21, path_length_21 = angle_of_turn(tang_pt_11, tang_pt_31, middle_circle_1, rad_turn, path_type.lower()[1])
        angle_turn_31, path_length_31 = angle_of_turn(tang_pt_31, fin_config, final_circle, rad_turn, path_type.lower()[2])
        # Path number 2
        angle_turn_12, path_length_12 = angle_of_turn(ini_config, tang_pt_12, initial_circle, rad_turn, path_type.lower()[0])
        angle_turn_22, path_length_22 = angle_of_turn(tang_pt_12, tang_pt_32, middle_circle_2, rad_turn, path_type.lower()[1])
        angle_turn_32, path_length_32 = angle_of_turn(tang_pt_32, fin_config, final_circle, rad_turn, path_type.lower()[2])
        
        path_length = np.array([path_length_11 + path_length_21 + path_length_31,\
                                path_length_12 + path_length_22 + path_length_32])
        
        if visualization == 1:
            
            plt.figure()
            # Plotting the first path
            # Plotting the initial and final configurations
            plt.arrow(ini_config[0], ini_config[1], math.cos(ini_config[2]), math.sin(ini_config[2]),\
                  length_includes_head = True, head_width = .2, color = 'orange', label = 'Initial configuration')
            plt.arrow(fin_config[0], fin_config[1], math.cos(fin_config[2]), math.sin(fin_config[2]),\
                  length_includes_head = True, head_width = .2, color = 'black', label = 'Final configuration')
            # Plotting the initial, middle, and final circles
            circle1 = plt.Circle((initial_circle[0], initial_circle[1]), rad_turn,\
                                      color = 'r', fill = False, label = 'Initial circle')
            plt.gca().add_patch(circle1)
            circle2 = plt.Circle((middle_circle_1[0], middle_circle_1[1]), rad_turn,\
                                  color = 'brown', fill = False, label = 'Middle circle')
            plt.gca().add_patch(circle2)
            circle3 = plt.Circle((final_circle[0], final_circle[1]), rad_turn,\
                                  color = 'g', fill = False, label = 'Final circle')
            plt.gca().add_patch(circle3)
            
            # Path on the first circle
            # Range for the heading angle
            theta_1 = np.linspace(ini_config[2], ini_config[2] + angle_turn_11, 100)
            # Points on the first circle
            # NOTE: theta_1 provides discretization of the headings as the Dubins vehicle moves from the initial
            # location to the tangent point. The angle traced on the circle, which is on the left or right of the initial
            # location is equal to the instanteous heading minus or plus pi/2, respectivley. This is because the vector
            # connecting the center of the circle with the instaneous configuration has a phase angle equal to the
            # instaneous heading minus (for left turn) or plus (for right turn) pi/2.
            if path_type.lower()[0] == 'l':
            
                pts_circle_1_x_coord = np.array([initial_circle[0] + rad_turn*math.cos(i - math.pi/2) for i in theta_1])
                pts_circle_1_y_coord = np.array([initial_circle[1] + rad_turn*math.sin(i - math.pi/2) for i in theta_1])
                # Angle of the vector connecting the center of the middle circle with the tangent point
                angle_tang_pt_circle_2 = math.pi + theta_1[-1] - math.pi/2
                
            else:
                
                pts_circle_1_x_coord = np.array([initial_circle[0] + rad_turn*math.cos(i + math.pi/2) for i in theta_1])
                pts_circle_1_y_coord = np.array([initial_circle[1] + rad_turn*math.sin(i + math.pi/2) for i in theta_1])
                # Angle of the vector connecting the center of the middle circle with the tangent point
                angle_tang_pt_circle_2 = math.pi + theta_1[-1] + math.pi/2
                
            # Path on the second circle
            # Range of the angle of the points on the middle circle
            # NOTE: This is the range of angle made by the vector connecting the center of the middle circle with the
            # poins on the path on the middle circle. Therefore, we don't require an adjustment depending on whether
            # a left turn or right turn is being made (like what was done with theta_1)
            theta_2 = np.linspace(angle_tang_pt_circle_2, angle_tang_pt_circle_2 + angle_turn_21, 100)
            # Points on the second circle
            pts_circle_2_x_coord = np.array([middle_circle_1[0] + rad_turn*math.cos(i) for i in theta_2])
            pts_circle_2_y_coord = np.array([middle_circle_1[1] + rad_turn*math.sin(i) for i in theta_2])
            
            # Path on the third circle
            theta_3 = np.linspace(fin_config[2] - angle_turn_31, fin_config[2], 100)
            # Same note as that for theta_3            
            if path_type.lower()[2] == 'l':
                
                pts_circle_3_x_coord = np.array([final_circle[0] + rad_turn*math.cos(i - math.pi/2) for i in theta_3])
                pts_circle_3_y_coord = np.array([final_circle[1] + rad_turn*math.sin(i - math.pi/2) for i in theta_3])
            
            else:
                
                pts_circle_3_x_coord = np.array([final_circle[0] + rad_turn*math.cos(i + math.pi/2) for i in theta_3])
                pts_circle_3_y_coord = np.array([final_circle[1] + rad_turn*math.sin(i + math.pi/2) for i in theta_3])
            
            # x coordinates for the complete path
            x_coords_path = np.append(np.append(pts_circle_1_x_coord, pts_circle_2_x_coord), pts_circle_3_x_coord)
            # y coordinates for the complete path
            y_coords_path = np.append(np.append(pts_circle_1_y_coord, pts_circle_2_y_coord), pts_circle_3_y_coord)
            # Plotting the complete path
            plt.plot(x_coords_path, y_coords_path, color = 'blue', label = path_type.upper() + ' path 1')
            
            # Plotting the centers of the three circles
            plt.scatter([initial_circle[0], middle_circle_1[0], final_circle[0]],\
                        [initial_circle[1], middle_circle_1[1], final_circle[1]], marker = '*')
            
            plt.legend()
            
            plt.show()
            
            plt.figure()
            # Plotting the second path
            # Plotting the initial and final configurations
            plt.arrow(ini_config[0], ini_config[1], math.cos(ini_config[2]), math.sin(ini_config[2]),\
                  length_includes_head = True, head_width = .2, color = 'orange', label = 'Initial configuration')
            plt.arrow(fin_config[0], fin_config[1], math.cos(fin_config[2]), math.sin(fin_config[2]),\
                  length_includes_head = True, head_width = .2, color = 'black', label = 'Final configuration')
            # Plotting the initial, middle, and final circles
            circle1 = plt.Circle((initial_circle[0], initial_circle[1]), rad_turn,\
                                      color = 'r', fill = False, label = 'Initial circle')
            plt.gca().add_patch(circle1)
            circle2 = plt.Circle((middle_circle_2[0], middle_circle_2[1]), rad_turn,\
                                  color = 'brown', fill = False, label = 'Middle circle')
            plt.gca().add_patch(circle2)
            circle3 = plt.Circle((final_circle[0], final_circle[1]), rad_turn,\
                                  color = 'g', fill = False, label = 'Final circle')
            plt.gca().add_patch(circle3)
            
            # Path on the first circle
            # Range for the heading angle
            theta_1 = np.linspace(ini_config[2], ini_config[2] + angle_turn_12, 100)
            # Points on the first circle
            # NOTE: theta_1 provides discretization of the headings as the Dubins vehicle moves from the initial
            # location to the tangent point. The angle traced on the circle, which is on the left or right of the initial
            # location is equal to the instanteous heading minus or plus pi/2, respectivley. This is because the vector
            # connecting the center of the circle with the instaneous configuration has a phase angle equal to the
            # instaneous heading minus (for left turn) or plus (for right turn) pi/2.
            if path_type.lower()[0] == 'l':
            
                pts_circle_1_x_coord = np.array([initial_circle[0] + rad_turn*math.cos(i - math.pi/2) for i in theta_1])
                pts_circle_1_y_coord = np.array([initial_circle[1] + rad_turn*math.sin(i - math.pi/2) for i in theta_1])
                # Angle of the vector connecting the center of the middle circle with the tangent point
                angle_tang_pt_circle_2 = math.pi + theta_1[-1] - math.pi/2
                
            else:
                
                pts_circle_1_x_coord = np.array([initial_circle[0] + rad_turn*math.cos(i + math.pi/2) for i in theta_1])
                pts_circle_1_y_coord = np.array([initial_circle[1] + rad_turn*math.sin(i + math.pi/2) for i in theta_1])
                # Angle of the vector connecting the center of the middle circle with the tangent point
                angle_tang_pt_circle_2 = math.pi + theta_1[-1] + math.pi/2
                
            # Path on the second circle
            # Range of the angle of the points on the middle circle
            # NOTE: This is the range of angle made by the vector connecting the center of the middle circle with the
            # poins on the path on the middle circle. Therefore, we don't require an adjustment depending on whether
            # a left turn or right turn is being made (like what was done with theta_1)
            theta_2 = np.linspace(angle_tang_pt_circle_2, angle_tang_pt_circle_2 + angle_turn_22, 100)
            # Points on the second circle
            pts_circle_2_x_coord = np.array([middle_circle_2[0] + rad_turn*math.cos(i) for i in theta_2])
            pts_circle_2_y_coord = np.array([middle_circle_2[1] + rad_turn*math.sin(i) for i in theta_2])
            
            # Path on the third circle
            theta_3 = np.linspace(fin_config[2] - angle_turn_32, fin_config[2], 100)
            # Same note as that for theta_3            
            if path_type.lower()[2] == 'l':
                
                pts_circle_3_x_coord = np.array([final_circle[0] + rad_turn*math.cos(i - math.pi/2) for i in theta_3])
                pts_circle_3_y_coord = np.array([final_circle[1] + rad_turn*math.sin(i - math.pi/2) for i in theta_3])
            
            else:
                
                pts_circle_3_x_coord = np.array([final_circle[0] + rad_turn*math.cos(i + math.pi/2) for i in theta_3])
                pts_circle_3_y_coord = np.array([final_circle[1] + rad_turn*math.sin(i + math.pi/2) for i in theta_3])
            
            # x coordinates for the complete path
            x_coords_path = np.append(np.append(pts_circle_1_x_coord, pts_circle_2_x_coord), pts_circle_3_x_coord)
            # y coordinates for the complete path
            y_coords_path = np.append(np.append(pts_circle_1_y_coord, pts_circle_2_y_coord), pts_circle_3_y_coord)
            # Plotting the complete path
            plt.plot(x_coords_path, y_coords_path, color = 'blue', label = path_type.upper() + ' path 2')
            
            # Plotting the centers of the three circles
            plt.scatter([initial_circle[0], middle_circle_2[0], final_circle[0]],\
                        [initial_circle[1], middle_circle_2[1], final_circle[1]], marker = '*')
            
            plt.legend()
            
            plt.show()
        
    return path_length, initial_circle, final_circle, middle_circle_1, tang_pt_11, tang_pt_31,\
        angle_turn_11, angle_turn_21, angle_turn_31, middle_circle_2, tang_pt_12, tang_pt_32,\
        angle_turn_12, angle_turn_22, angle_turn_32
    
#%% Alternate implementation of Dubins path
    
def ini_fin_config_manipulate(ini_config, fin_config):
    '''
    This function translates the initial configuration to the origin and updates
    the final configuration. The headings are also modified such that the final
    configuration is along the x-axis.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the coordinates of the initial position and the heading angle.
    fin_config : Numpy 1x3 array
        Contains the coordinates of the final position and the heading angle.

    Returns
    -------
    ini_config_mod : Numpy 1x3 array
        Contains the modified coordinates for the initial position such that it
        coincides with the origin, and the modified heading such that the final
        position is along the x-axis.
    fin_config_mod : Numpy 1x3 array
        Contains the modified coordinates for the initial position such that it
        coincides with the origin, and the modified heading such that the final
        position is along the x-axis.
    d : Scalar
        Eucledian distance between the initial and final positions
    opt_pt : Boolean (if opt_pt_mod is False) or Numpy 1x2 array
        Contains the coordinates of the point in the original frame of reference.
    
    '''
    
    # Angle of vector connecting initial and final configurations
    ang_vec = math.atan2((fin_config[1] - ini_config[1]),\
                         (fin_config[0] - ini_config[0]))
        
    # Length of the vector connecting the initial and final configurations
    d = math.sqrt((fin_config[0] - ini_config[0])**2 +\
                  (fin_config[1] - ini_config[1])**2)
    
    # Modified initial and final configurations
    ini_config_mod = np.array([0, 0, ini_config[2] - ang_vec])
    fin_config_mod = np.array([fin_config[0] - ini_config[0],\
                               fin_config[1] - ini_config[1],\
                               fin_config[2] - ang_vec])
        
    return ini_config_mod, fin_config_mod, d
    
def Seg_pts(start_pt_config, length_seg, rad_turn, seg_type = 's'):
    
    '''
    This function returns an array of points corresponding to a left turn,
    right turn, or straight line segment, given the configuration corresponding
    to the start of the segment, the initial and final configurations in the original
    frame of reference, the arc length for the segment, the radius of the tight turn
    (used for left and right tight turns), and the type of the segment.
    The points returned are in the origin frame of reference.

    Parameters
    ----------
    start_pt_config : Numpy 1x3 array
        Configuration of the start of the arc in the original frame of reference.
    length_seg : Scalar
        Length of the segment.
    rad_turn : Scalar
        Radius of the tight turn.
    seg_type : Character
        's' - straight line segment
        'l' - left tight turn, whose radius is given by rad_tight_turn
        'r' - right tight turn, whose radius if given by rad_tight_turn

    Returns
    -------
    pts_original_frame : Numpy nx3 array
        Contains the coordinates of the points along the left segment turn.
    
    '''
    
    # Discretizing the segment length
    segment_disc = np.linspace(0, length_seg, 100)
    pts_original_frame = np.zeros((100, 3))
    
    if seg_type.lower() == 'l' or seg_type.lower() == 'r':
        
        # # Discretizing the segment length
        # segment_disc = np.linspace(0, length_seg, 100)
        # pts_original_frame = np.zeros((100, 3))
    
        # Finding the coordinates of the point corresponding to the discretization
        # of the segment if left or right turn
        for i in range(len(segment_disc)):
            
            if seg_type.lower() == 'l':
        
                # x-coordinate
                pts_original_frame[i, 0] = start_pt_config[0]\
                    + rad_turn*math.sin(start_pt_config[2] + segment_disc[i]/rad_turn)\
                    - rad_turn*math.sin(start_pt_config[2])
                # y-coordinate
                pts_original_frame[i, 1] = start_pt_config[1]\
                    - rad_turn*math.cos(start_pt_config[2] + segment_disc[i]/rad_turn)\
                    + rad_turn*math.cos(start_pt_config[2])
                # Heading
                pts_original_frame[i, 2] = start_pt_config[2] + segment_disc[i]/rad_turn
                
            elif seg_type.lower() == 'r':
        
                # x-coordinate
                pts_original_frame[i, 0] = start_pt_config[0]\
                    - rad_turn*math.sin(start_pt_config[2] - segment_disc[i]/rad_turn)\
                    + rad_turn*math.sin(start_pt_config[2])
                # y-coordinate
                pts_original_frame[i, 1] = start_pt_config[1]\
                    + rad_turn*math.cos(start_pt_config[2] - segment_disc[i]/rad_turn)\
                    - rad_turn*math.cos(start_pt_config[2])
                # Heading
                pts_original_frame[i, 2] = start_pt_config[2] - segment_disc[i]/rad_turn
        
    elif seg_type.lower() == 's':
        
        # PREVIOUS IMPLEMENTATION: JUST GET THE INITIAL AND FINAL POINT ON THE
        # LINE. WHILE THIS WORKS FOR WHEN WE PLOT ON THE PLANE, THIS CAUSES AN
        # ISSUE WHEN WRAPPED ONTO THE CYLINDER. THEREFORE, WE INSTEAD GENERATE
        # NUMEROUS POINTS ALONG THE LINE, AS THIS WOULD AVERT THE ISSUE WHEN THE
        # LINE IS WRAPPED ONTO THE CYLINDER.
        
        # pts_original_frame = np.zeros((2, 3))
        # # First point on the straight line segment is the start point of the segment itself
        # pts_original_frame[0, :] = start_pt_config
        # # Second point on the straight line segment
        # pts_original_frame[1, :] = np.array([start_pt_config[0] + length_seg*math.cos(start_pt_config[2]),\
        #                                      start_pt_config[1] + length_seg*math.sin(start_pt_config[2]),\
        #                                      start_pt_config[2]])
        
        # Finding the coordinates of the point corresponding to the discretization
        # on the line
        for i in range(len(segment_disc)):
            
            # x-coordinate
            pts_original_frame[i, 0] = start_pt_config[0] +\
                segment_disc[i]*math.cos(start_pt_config[2])
            # y-coordinate
            pts_original_frame[i, 1] = start_pt_config[1] +\
                segment_disc[i]*math.sin(start_pt_config[2])
            # Heading
            pts_original_frame[i, 2] = start_pt_config[2]
        
    return pts_original_frame

def points_path(ini_config, fin_config, rad_turn, t, p, q, path_type):
    '''
    This function returns points along the path in the original frame of reference
    given the initial and final configurations, the radius of turn, the length of
    each segment of the path, and the path type

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    rad_turn : Scalar
        Radius of the tight turn.
    t : Scalar
        Length of the first segment of the path.
    p : Scalar
        Length of the second segment of the path.
    q : Scalar
        Length of the third segment of the path.
    path_type : String
        Contains the path type, and contains three segments made up of 'l' for left turn,
        'r' for right turn, and 's' for straight line segment.

    Returns
    -------
    x_coords_path : Numpy nx3 array
        Contains the x-coordinate of points generated along the path.
    y_coords_path : Numpy nx3 array
        Contains the y-coordinate of points generated along the path.

    '''
    
    # Obtaining the points along the path
    # pts_first_seg = Seg_pts(ini_config, ini_config, fin_config, t, rad_turn, path_type[0])
    # pts_second_seg = Seg_pts(pts_first_seg[-1], ini_config, fin_config, p, rad_turn, path_type[1])
    # pts_third_seg = Seg_pts(pts_second_seg[-1], ini_config, fin_config, q, rad_turn, path_type[2])
    pts_first_seg = Seg_pts(ini_config, t, rad_turn, path_type[0])
    pts_second_seg = Seg_pts(pts_first_seg[-1], p, rad_turn, path_type[1])
    pts_third_seg = Seg_pts(pts_second_seg[-1], q, rad_turn, path_type[2])
    # x coordinates for the complete path in the original frame of reference
    x_coords_path = np.append(np.append(pts_first_seg[:, 0], pts_second_seg[:, 0]), pts_third_seg[:, 0])
    # y coordinates for the complete path in the original frame of reference
    y_coords_path = np.append(np.append(pts_first_seg[:, 1], pts_second_seg[:, 1]), pts_third_seg[:, 1])
    return x_coords_path, y_coords_path

def CSC_path_efficient(ini_config, fin_config, rad_turn, path_type = 'lsl', visualization = 1):
    '''
    This function generates a CSC path connecting the initial and final configurations.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    rad_turn : Scalar
        Radius of the tight turn.
    path_type : String
        Contains the path type, and contains three segments made up of 'l' for left turn,
        'r' for right turn, and 's' for straight line segment.
    visualization : Scalar
        If equal to 1, a plot depicting the path is generated.

    Returns
    -------
    path_length : Scalar
        Length of the path.
    t : Scalar
        Length of the first segment of the path.
    p : Scalar
        Length of the second segment of the path.
    q : Scalar
        Length of the third segment of the path.

    '''
    
    # Initial and final configurations after modification, i.e., shifting the
    # initial configuration to the origin and the final configuration along
    # the x-axis
    ini_config_mod, fin_config_mod, d = ini_fin_config_manipulate(ini_config, fin_config)
    
    # Initial and final heading angles in the modified frame of reference
    alpha = ini_config_mod[2]
    beta = fin_config_mod[2]
    # Modified distance between initial and final configurations based on the value
    # for the radius of the tight turn
    ddash = d/rad_turn
    
    # Square of length of straight line path    
    if path_type.lower() == 'lsl':
    
        ptemp = (rad_turn**2)*(2 + ddash**2 - 2*math.cos(alpha - beta)\
                             + 2*ddash*(math.sin(alpha) - math.sin(beta)))
        
    elif path_type.lower() == 'rsr':
        
        ptemp = (rad_turn**2)*(2 + ddash**2 - 2*math.cos(alpha - beta)\
                             + 2*ddash*(math.sin(beta) - math.sin(alpha)))
            
    elif path_type.lower() == 'lsr':
        
        ptemp = (rad_turn**2)*(ddash**2 - 2 + 2*math.cos(alpha - beta)\
                             + 2*ddash*(math.sin(beta) + math.sin(alpha)))
            
    elif path_type.lower() == 'rsl':
        
        ptemp = (rad_turn**2)*(ddash**2 - 2 + 2*math.cos(alpha - beta)\
                             - 2*ddash*(math.sin(beta) + math.sin(alpha)))
    
    # Checking if the path exists: p is the value of a function under
    if path_type[1].lower() == 's' and ptemp < 0:
        
        print(path_type.upper() + ' path does not exist.')
        path_length = t = p = q = np.NaN
        
    else:
        
        if ptemp == 0 and path_type[0].lower() == path_type[2].lower():
        
            print('Path is of type ' + path_type[0].upper() + '.')
            
        elif ptemp == 0:
            
            print('Path is of type ' + path_type[0].upper() + path_type[2].upper() + '.')
            
        # Length of straight line segment
        p = math.sqrt(ptemp)
        
        # Computing the length of the first and last arcs
        if path_type[0].lower() == path_type[2].lower() == 'l':
            
            # Length of first arc
            t = rad_turn*(np.mod(math.atan2(math.cos(beta) - math.cos(alpha),\
                                            ddash + math.sin(alpha) - math.sin(beta))\
                                  - alpha, 2*math.pi))
            # Length of final arc
            q = rad_turn*(np.mod(beta - math.atan2(math.cos(beta) - math.cos(alpha),\
                                                   ddash + math.sin(alpha) - math.sin(beta)),\
                                 2*math.pi))
            
        elif path_type[0].lower() == path_type[2].lower() == 'r':
            
            # Length of first arc
            t = rad_turn*(np.mod(-math.atan2(math.cos(alpha) - math.cos(beta),\
                                             math.sin(beta) - math.sin(alpha) + ddash)\
                                 + alpha, 2*math.pi))
            # Length of final arc
            q = rad_turn*(np.mod(-beta + math.atan2(math.cos(alpha) - math.cos(beta),\
                                                    math.sin(beta) - math.sin(alpha) + ddash),\
                                 2*math.pi))
                
        elif path_type[0].lower() == 'l' and path_type[2].lower() == 'r':
            
            # Length of first arc
            t = rad_turn*(np.mod(-alpha + math.atan2(2*rad_turn, p)\
                                 + math.atan2(-math.cos(beta) - math.cos(alpha),\
                                              ddash + math.sin(beta) + math.sin(alpha)),\
                                 2*math.pi))
            # Length of final arc
            q = rad_turn*(np.mod(-beta + math.atan2(2*rad_turn, p)\
                                 + math.atan2(-math.cos(beta) - math.cos(alpha),\
                                              ddash + math.sin(beta) + math.sin(alpha)),\
                                 2*math.pi))
                
        elif path_type[0].lower() == 'r' and path_type[2].lower() == 'l':
            
            # Length of first arc
            t = rad_turn*(np.mod(alpha + math.atan2(2*rad_turn, p)\
                                 - math.atan2(math.cos(beta) + math.cos(alpha),\
                                              ddash - math.sin(beta) - math.sin(alpha)),\
                                 2*math.pi))
            # Length of final arc
            q = rad_turn*(np.mod(beta + math.atan2(2*rad_turn, p)\
                                 - math.atan2(math.cos(beta) + math.cos(alpha),\
                                              ddash - math.sin(beta) - math.sin(alpha)),\
                                 2*math.pi))
        # CHECK MORE CONDITIONS
        
        path_length = p + t + q
        
        # Plotting the path
        if visualization == 1:
            
            plotting_path(ini_config, fin_config, rad_turn, t, p, q, path_type.lower())
    
    return path_length, t, p, q

def CCC_path_efficient(ini_config, fin_config, rad_turn, path_type = 'lrl', visualization = 1):
    '''
    This function generates a CSC path connecting the initial and final configurations.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    rad_turn : Scalar
        Radius of the tight turn.
    path_type : String
        Contains the path type, and contains three segments made up of 'l' for left turn,
        'r' for right turn, and 's' for straight line segment.
    visualization : Scalar
        If equal to 1, a plot depicting the path is generated.

    Returns
    -------
    path_length : Scalar or Numpy 1x2 array
        Length of the path. Scalar (NaN) returned if path does not exist, whereas
        numpy array with two path lengths (due to two middle circles possible) returned.
    t : Scalar or Numpy 1x2 array
        Length of the first segment of the path. Scalar (NaN) returned if path does
        not exist, whereas numpy array with two path lengths (due to two middle
        circles possible) returned.
    p : Scalar or Numpy 1x2 array
        Length of the second segment of the path. Scalar (NaN) returned if path does
        not exist, whereas numpy array with two path lengths (due to two middle
        circles possible) returned.
    q : Scalar or Numpy 1x2 array
        Length of the third segment of the path. Scalar (NaN) returned if path does
        not exist, whereas numpy array with two path lengths (due to two middle
        circles possible) returned.

    '''
    # Initial and final configurations after modification, i.e., shifting the
    # initial configuration to the origin and the final configuration along
    # the x-axis
    ini_config_mod, fin_config_mod, d = ini_fin_config_manipulate(ini_config, fin_config)
    
    # Initial and final heading angles in the modified frame of reference
    alpha = ini_config_mod[2]
    beta = fin_config_mod[2]
    # Modified distance between initial and final configurations based on the value
    # for the radius of the tight turn
    ddash = d/rad_turn
    
    # Length of middle arc           
    if path_type.lower() == 'lrl':
        
        ptemp = 0.125*(6 - ddash**2 + 2*ddash*(math.sin(beta) - math.sin(alpha))\
                        + 2*math.cos(alpha - beta))
        
    elif path_type.lower() == 'rlr':
        
        ptemp = 0.125*(6 - ddash**2 - 2*ddash*(math.sin(beta) - math.sin(alpha))\
                        + 2*math.cos(alpha - beta))
        
    # Checking if the path exists if CCC type: p is the value of an arccos func
    if abs(ptemp) > 1:
        
        print(path_type.upper() + ' path does not exist.')
        path_length = t = p = q = np.NaN
        
    else:
        
        if ptemp == 1:
            
            print('Path is of type ', path_type[0].upper())
            
        # Length of middle arc for one paths if ptemp != 1 or -1
        if abs(ptemp) == 1:
            
            p = rad_turn*np.array([math.acos(ptemp)])
         
        # Two paths exist if -1 < ptemp < 1 
        else:
            
            p = rad_turn*np.array([math.acos(ptemp), 2*math.pi - math.acos(ptemp)])
            
        # Finding the length of the first and last arcs for one path if
        # ptemp = +-1, and for two paths if -1 < ptemp < 1            
        if path_type[0].lower() == 'l':
            
            # Length of first arc
            t = np.array([rad_turn*(np.mod(-alpha + i/(2*rad_turn)\
                                           + math.atan2(math.cos(beta) - math.cos(alpha),\
                                                        ddash - math.sin(beta) + math.sin(alpha)),\
                                           2*math.pi)) for i in p])
            # Length of last arc
            q = np.array([rad_turn*(np.mod(beta + i/(2*rad_turn)\
                                            - math.atan2(math.cos(beta) - math.cos(alpha),\
                                                        ddash - math.sin(beta) + math.sin(alpha)),\
                                            2*math.pi)) for i in p])
            
        else:
            
            # Length of first arc
            t = np.array([rad_turn*(np.mod(alpha + i/(2*rad_turn)\
                                           - math.atan2(math.cos(alpha) - math.cos(beta),\
                                                        ddash - math.sin(alpha) + math.sin(beta)),\
                                           2*math.pi)) for i in p])
            # Length of last arc
            q = np.array([rad_turn*(np.mod(-beta + i/(2*rad_turn)\
                                           + math.atan2(math.cos(alpha) - math.cos(beta),\
                                                        ddash - math.sin(alpha) + math.sin(beta)),\
                                           2*math.pi)) for i in p])
        
        path_length = p + t + q
        
        # Plotting the path
        if visualization == 1:
            
            for i in range(len(p)):
                
                name_plot = path_type.lower() + str(i + 1)
                plotting_path(ini_config, fin_config, rad_turn, t[i], p[i], q[i], name_plot)
        
    return path_length, t, p, q

def plotting_path(ini_config, fin_config, rad_turn, t, p, q, path_type):
    '''
    This function plots a path connecting the initial and final configurations
    along a required path type.

    Parameters
    ----------
    ini_config : Numpy 1x3 array
        Contains the initial configuration.
    fin_config : Numpy 1x3 array
        Contains the final configuration.
    rad_turn : Scalar
        Radius of the tight turn.
    t : Scalar
        Length of the first segment of the path.
    p : Scalar
        Length of the second segment of the path.
    q : Scalar
        Length of the third segment of the path.
    path_type : String
        Contains the path type, and contains three segments made up of 'l' for left turn,
        'r' for right turn, and 's' for straight line segment.

    Returns
    -------
    None.

    '''
    
    plt.figure()
    # Plotting the initial and final configurations
    plt.arrow(ini_config[0], ini_config[1], math.cos(ini_config[2]), math.sin(ini_config[2]),\
          length_includes_head = True, head_width = .2, color = 'orange', label = 'Initial configuration')
    plt.arrow(fin_config[0], fin_config[1], math.cos(fin_config[2]), math.sin(fin_config[2]),\
          length_includes_head = True, head_width = .2, color = 'black', label = 'Final configuration')
    
    # Obtaining the coordinates of points along the path
    x_coords_path, y_coords_path = points_path(ini_config, fin_config, rad_turn, t, p, q, path_type)
    
    # Plotting the complete path in the original frame of reference
    plt.plot(x_coords_path, y_coords_path, color = 'blue', label = path_type.upper() + ' path')
    plt.legend()