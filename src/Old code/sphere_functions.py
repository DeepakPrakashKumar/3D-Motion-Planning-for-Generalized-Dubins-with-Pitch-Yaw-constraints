import numpy as np
import os
import math

# Importing the plotting commands
path_plotting_class = 'D:\TAMU\Research\Cylinder code'
os.chdir(path_plotting_class)
# cwd = os.getcwd()
# print("Current working directory: {0}".format(cwd))
from plotting_class import plotting_functions
# import plotting_class
# Importing the sphere plotting command
path_plotting_class = 'D:\TAMU\Research\Cylinder and sphere'
os.chdir(path_plotting_class)
from cylinder_sphere_functions import generate_points_sphere

# Modifying path for the current code
path = 'D:\TAMU\Research\Cylinder and sphere'
os.chdir(path)

def random_config_sphere(R):
    """This function generates a random configuration on a sphere of radius R
     located at the origin.

    Args:
        R (Scalar): Radius of the sphere

    Returns:
        config (Numpy 3x3 array): Contains the position (in first row), tangent
         vector (in second row),
         and normal vector (in third array).
    """

    # Generating random numbers for the angles on the sphere
    phi = np.random.rand(1)*2*math.pi
    theta = np.random.rand(1)*2*math.pi

    # Position on the sphere - unit vector
    pos = np.array([math.cos(theta)*math.cos(phi), math.cos(theta)*math.sin(phi),\
                    math.sin(theta)])

    # Generating a random vector and orthonormalizing it with respect to the position vector
    temp = np.random.rand(3)
    if np.linalg.norm(temp - np.dot(pos, temp)*pos) > 0.01: # tolerance for checking linear independence

        T = (temp - np.dot(pos, temp)*pos)/np.linalg.norm(temp - np.dot(pos, temp)*pos)

    else:

        raise Exception("Regenerate the random vector")

    # Computing the normal vector as the cross product of the position and tangent vectors
    N = np.cross(pos, T)

    # Scaling the position vector based on the radius of the sphere
    pos = pos*R

    return np.array([pos, T, N])

def generate_points_seg(config, r, R, angle, type_seg = 'l'):
    '''
    This function generates points on a segment. The segment can be an arc of a
    great circle or an arc of a tight circle.

    Parameters
    ----------
    config : Numpy 3x3 array
        Contains the initial position, tangent vector, and normal vector on the sphere.
    r : Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the sphere.
    angle : Scalar
        Angle of the arc.
    type_seg : Character, optional
        Describes the type of the arc. 'g' - great circle arc, 'l' - left tight circle,
        'r' - right tight circle. The default is 'l'.

    Raises
    ------
    Exception
        Exception raised when incorrect segment type is passed.

    Returns
    -------
    points_seg : Numpy array
        Contains the coordinates on points on the path.

    '''
    
    phi = np.linspace(0, angle, 100)
    points_seg = np.empty((100, 3))
    
    if type_seg.lower() == 'g':
        
        for i in range(len(phi)):
            
            points_seg[i] = math.cos(phi[i])*config[0, :] + R*math.sin(phi[i])*config[1, :]
        
    elif type_seg.lower() == 'l':
        
        for i in range(len(phi)):
            
            points_seg[i] = (1 - (1 - math.cos(phi[i]))*(r/R)**2)*config[0, :]\
                + r*math.sin(phi[i])*config[1, :]\
                + (1 - math.cos(phi[i]))*r*math.sqrt(1 - (r/R)**2)*config[2, :]
        
    elif type_seg.lower() == 'r':
        
        for i in range(len(phi)):
            
            points_seg[i] = (1 - (1 - math.cos(phi[i]))*(r/R)**2)*config[0, :]\
                + r*math.sin(phi[i])*config[1, :]\
                - (1 - math.cos(phi[i]))*r*math.sqrt(1 - (r/R)**2)*config[2, :]
        
    else:
        
        raise Exception("Incorrect string")
    
    return points_seg

def test_fun_path(r, R, angle):
    
    ini_config = random_config_sphere(R)
    points_left_turn = generate_points_seg(ini_config, r, R, angle, 'l')
    points_right_turn = generate_points_seg(ini_config, r, R, angle, 'r')
    points_great_circle_turn = generate_points_seg(ini_config, r, R, angle, 'g')
    
    fig = plotting_functions()
    # Adding the sphere
    x_sp, y_sp, z_sp = generate_points_sphere(np.array([0, 0, 0]), R)
    fig.surface_3D(x_sp, y_sp, z_sp, 'grey', 'Surface', 0.4)    
    # Adding the configuration
    fig.points_3D([ini_config[0, 0]], [ini_config[0, 1]], [ini_config[0, 2]], 'red',
                  'Initial configuration')
    # Adding the tangent and normal vectors
    fig.arrows_3D([ini_config[0, 0]], [ini_config[0, 1]], [ini_config[0, 2]],\
                  [ini_config[1, 0]], [ini_config[1, 1]], [ini_config[1, 2]],
                  'orange', 'oranges', 'Initial tangent vector', 5, 15, 5)
    fig.arrows_3D([ini_config[0, 0]], [ini_config[0, 1]], [ini_config[0, 2]],\
                  [ini_config[2, 0]], [ini_config[2, 1]], [ini_config[2, 2]],
                  'brown', 'brwnyl', 'Initial normal vector', 5, 15, 5)
    # Adding the paths
    fig.scatter_3D(points_left_turn[:, 0], points_left_turn[:, 1], points_left_turn[:, 2],\
                   'blue', 'Left turn')
    fig.scatter_3D(points_right_turn[:, 0], points_right_turn[:, 1], points_right_turn[:, 2],\
                   'orange', 'Right turn')
    fig.scatter_3D(points_great_circle_turn[:, 0], points_great_circle_turn[:, 1],\
                   points_great_circle_turn[:, 2], 'black', 'Great circle arc')
        
    # Updating the figure and writing onto html file
    fig.update_layout_3D('X (m)', 'Y (m)', 'Z (m)', 'Segments on sphere')
    fig.writing_fig_to_html('test.html', 'w')