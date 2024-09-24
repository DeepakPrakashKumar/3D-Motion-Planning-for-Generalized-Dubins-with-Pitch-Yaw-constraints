# This is the main function to be called for generating a feasible solution connecting the initial and final configurations
# by connecting osculating spheres with same directionality by a cylindrical envelope, and osculating spheres with different
# directionality by conical envelopes.

import math
from main_functions import generate_random_configs_3D

# Here, we provide the description of the initial and final configurations if we
# want to provide the initial and final configurations.


# If a random initial and final configuration ought to be generated, we randomly generate the initial and final configurations.
xlim = 40; ylim = 40; zlim = 40; # We provide the region in the 3d space wherein we want to generate the configuration.
# xlim is used to pick a random x coordinate between 0 and xlim; similar interpretation follows for ylim and zlim.
ini_config = generate_random_configs_3D(xlim, ylim, zlim)
fin_config = generate_random_configs_3D(xlim, ylim, zlim)

# We now provide the parameters for the vehicle
R = 10 # Radius of the osculating sphere
r = 5 # This is the radius of the tight turn for the vehicle. Alternately, a geodesic curvature bound can be provided below.
# kg = 1; r = R/math.sqrt(1 + kg**2)

# We provide the number of discretizations to be considered for the location and the heading angle
loc_disc = 5
heading_ang_disc = 3

Dubins_3D_numerical_path_on_surfaces(ini_config, fin_config, r, R, disc_no)