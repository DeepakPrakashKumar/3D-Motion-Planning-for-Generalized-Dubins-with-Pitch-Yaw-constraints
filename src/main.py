# This is the main function to be called for generating a feasible solution connecting the initial and final configurations
# by connecting osculating spheres with same directionality by a cylindrical envelope, and osculating spheres with different
# directionality by conical envelopes.

import os
import sys
from main_functions_heuristic import generate_random_configs_3D, Dubins_3D_numerical_path_on_surfaces

# Including the following command to ensure that python is able to find the relevant files afer changing directory
sys.path.insert(0, '')
# Obtaining the current directory
cwd = os.getcwd()

# Importing code for plotting
rel_path = '\Visualization'
os.chdir(cwd + rel_path)
from visualization_simulation import plot_trajectory

# Returning to initial directory
os.chdir(cwd)

# Here, we provide the description of the initial and final configurations if we
# want to provide the initial and final configurations. For this implementation, we instead generate random initial and final configurations.


# If a random initial and final configuration ought to be generated, we randomly generate the initial and final configurations.
xlim = 40; ylim = 40; zlim = 40; # We provide the region in the 3d space wherein we want to generate the configuration.
# xlim is used to pick a random x coordinate between 0 and xlim; similar interpretation follows for ylim and zlim.
ini_config = generate_random_configs_3D(xlim, ylim, zlim)
fin_config = generate_random_configs_3D(xlim, ylim, zlim)

# We now provide the parameters for the vehicle
R = 10 # Radius of the osculating sphere
r = 5 # This is the radius of the tight turn for the vehicle.

# We provide the number of discretizations to be considered for the location and the heading angle
disc_no = 5

# We call the main function that constructs feasible solutions through sphere-cylinder-sphere, sphere-plane-sphere, and sphere-sphere-sphere
# combinations, and returns the path and orientations corresponding to the best path.
min_dist_path_length, min_dist_path_pts, tang_global_path, tang_normal_global_path, surf_normal_global_path, path_type =\
      Dubins_3D_numerical_path_on_surfaces(ini_config, fin_config, r, R, disc_no, visualization = 1, filename = 'temp.html')

print('Tangent vectors for the path are:\n', tang_global_path)

# We now simulate the motion of a vehicle along the path that we have obtained.
plot_trajectory(ini_config, fin_config, min_dist_path_pts, tang_global_path, tang_normal_global_path, surf_normal_global_path, path_type, R)