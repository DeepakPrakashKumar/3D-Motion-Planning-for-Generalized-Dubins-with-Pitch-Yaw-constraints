# This is the main function to be called for generating a feasible solution connecting the initial and final configurations
# by connecting osculating spheres with same directionality by a cylindrical envelope, and osculating spheres with different
# directionality by conical envelopes.

import math
from main_functions_heuristic import generate_random_configs_3D, Dubins_3D_numerical_path_on_surfaces

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
Dubins_3D_numerical_path_on_surfaces(ini_config, fin_config, r, R, disc_no)