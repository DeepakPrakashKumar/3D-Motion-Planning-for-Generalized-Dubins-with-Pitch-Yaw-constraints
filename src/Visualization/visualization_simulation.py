import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import os
import sys
from msg_state import MsgState
import time

# # Including the following command to ensure that python is able to find the relevant files afer changing directory
# sys.path.insert(0, '')
# # Obtaining the current directory
# cwd = os.getcwd()

# # Importing code for the sphere
# rel_path = '\Sphere code'
# os.chdir(cwd + rel_path)
# from Path_generation_sphere import generate_points_sphere

# # Returning to initial directory
# os.chdir(cwd)

from view_manager import ViewManager

# The code for Arrow3D is taken from https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
# The code for the animation is based on https://matplotlib.org/stable/gallery/mplot3d/wire3d_animation_sgskip.html
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)
    
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
    return arrow

setattr(Axes3D, 'arrow3D', _arrow3D)

def plot_trajectory(ini_config, fin_config, pos_global, tang_global_path, tang_normal_global_path,\
                     surf_normal_global_path, path_type, R):
    # In this function, the trajectory is visualized and is animated.

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Setting up an object for visualizing the aircraft
    viewers = ViewManager(ax, animation=True, video=False, video_name = 'trajectory_aircraft.mp4')

    # We define the length of the arrow for representing the orientation
    length = 3

    ax.scatter(ini_config[0, 0], ini_config[0, 1], ini_config[0, 2], marker = 'o', linewidth = 1.5,\
            color = 'r', label = 'Initial point')
    ax.scatter(fin_config[0, 0], fin_config[0, 1], fin_config[0, 2], marker = 'D', linewidth = 1.5,\
            color = 'b', label = 'Final point')
    # We plot the orientation of the vehicle as well
    ax.arrow3D(ini_config[0, 0], ini_config[0, 1], ini_config[0, 2], length*ini_config[1, 0],\
                length*ini_config[1, 1], length*ini_config[1, 2], mutation_scale=20, fc='red', label = 'Tangent vector')
    ax.arrow3D(ini_config[0, 0], ini_config[0, 1], ini_config[0, 2], length*ini_config[2, 0],\
                length*ini_config[2, 1], length*ini_config[2, 2], mutation_scale=20, fc='blue', label = 'Tangent normal vector')
    ax.arrow3D(ini_config[0, 0], ini_config[0, 1], ini_config[0, 2], length*ini_config[3, 0],\
                length*ini_config[3, 1], length*ini_config[3, 2], mutation_scale=20, fc='green', label = 'Surface normal vector')
    ax.arrow3D(fin_config[0, 0], fin_config[0, 1], fin_config[0, 2], length*fin_config[1, 0],\
                length*fin_config[1, 1], length*fin_config[1, 2], mutation_scale=20, fc='red')
    ax.arrow3D(fin_config[0, 0], fin_config[0, 1], fin_config[0, 2], length*fin_config[2, 0],\
                length*fin_config[2, 1], length*fin_config[2, 2], mutation_scale=20, fc='blue')
    ax.arrow3D(fin_config[0, 0], fin_config[0, 1], fin_config[0, 2], length*fin_config[3, 0],\
                length*fin_config[3, 1], length*fin_config[3, 2], mutation_scale=20, fc='green')

    ax.plot3D(pos_global[:, 0], pos_global[:, 1], pos_global[:, 2], linewidth = 1.5, label = 'Trajectory')
    
    ax.set_xlabel('X', fontsize = 12)
    ax.set_ylabel('Y', fontsize = 12)
    ax.set_zlabel('Z', fontsize = 12)

    # We also plot the spheres at the initial configuration
    if path_type in ['cyc_inner', 'plane_inner_outer', 'spheres_inner']:
        xini_inner, yini_inner, zini_inner = generate_points_sphere([ini_config[0, i] + R*ini_config[3, i] for i in range(3)], R)
        ax.plot_surface(xini_inner, yini_inner, zini_inner, color = 'orange', alpha=0.2)
    else:
        xini_outer, yini_outer, zini_outer = generate_points_sphere([ini_config[0, i] - R*ini_config[3, i] for i in range(3)], R)
        ax.plot_surface(xini_outer, yini_outer, zini_outer, color = 'magenta', alpha=0.2)

    # We also plot the spheres at the final configuration
    if path_type in ['cyc_inner', 'plane_outer_inner', 'spheres_inner']:
        xfin_inner, yfin_inner, zfin_inner = generate_points_sphere([fin_config[0, i] + R*fin_config[3, i] for i in range(3)], R)
        ax.plot_surface(xfin_inner, yfin_inner, zfin_inner, color = 'orange', alpha=0.2)
    else:
        xfin_outer, yfin_outer, zfin_outer = generate_points_sphere([fin_config[0, i] - R*fin_config[3, i] for i in range(3)], R)
        ax.plot_surface(xfin_outer, yfin_outer, zfin_outer, color = 'magenta', alpha=0.2)

    ax.legend(fontsize = 9, loc = 1)

    # We now begin plotting the configuration
    locplot = None; tangplot = None; surfplot = None; tangnorm_plot = None
    true_state = MsgState()
    for i in range(1, len(pos_global[:, 0])):

        # If a plot already exists, we remove it
        if locplot:
            locplot.remove()
            tangplot.remove()
            tangnorm_plot.remove()
            surfplot.remove()
            
        # We plot the current configuration
        locplot = ax.scatter(pos_global[i, 0], pos_global[i, 1], pos_global[i, 2], marker = 'o', linewidth = 1.5,\
            color = 'k')
        tangplot = ax.arrow3D(pos_global[i, 0], pos_global[i, 1], pos_global[i, 2], length*tang_global_path[i, 0],\
                length*tang_global_path[i, 1], length*tang_global_path[i, 2], mutation_scale=20, fc='red')
        tangnorm_plot = ax.arrow3D(pos_global[i, 0], pos_global[i, 1], pos_global[i, 2], length*tang_normal_global_path[i, 0],\
                    length*tang_normal_global_path[i, 1], length*tang_normal_global_path[i, 2], mutation_scale=20, fc='blue')
        surfplot = ax.arrow3D(pos_global[i, 0], pos_global[i, 1], pos_global[i, 2], length*surf_normal_global_path[i, 0],\
                    length*surf_normal_global_path[i, 1], length*surf_normal_global_path[i, 2], mutation_scale=20, fc='green')

        # We update the state of the aircraft
        true_state.north = pos_global[i, 0]
        true_state.east = pos_global[i, 1]
        true_state.altitude = pos_global[i, 2]

        # Plotting the aircraft
        viewers.update(
            time.time(),
            true_state = true_state,  # true states
        )

        # print('Printing the ', i, 'th point')
        plt.pause(.05)

    # We show the trajectory plot
    plt.show()

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