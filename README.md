# 3D Motion Planning for Generalized Dubins Vehicle considering Pitch and Yaw Rate Constraints

The repository contains the code for generating feasible solutions for the 3D Dubins problem using spheres, cylinders, and planes to connect an initial configuration to a final configuration.

# Main code

The main script for running the path planning algorithm is in src -> main.py. In this script, the initial and final configuration for the vehicle can be provided or randomly generated (in the script). In this model, since two control inputs are considered, the bounds for these two (Rpitch and Ryaw) must be provided as inputs. This script calls the function "Dubins_3D_numerical_path_on_surfaces", which contains the main implementation for the path construction algorithm. Additionally, the script also produces an animation for the vehicle traveling along the generated shortest path using the function "plot_trajectory".

(P.S. Stay tuned for additional detailed documentation!)

# Dependencies

pip install numpy-stl
pip install opencv-python
<!-- pip install pyqtgraph
pip install PyQt6 -->
pip3 install Pillow
pip install matplotlib (tested on versions 3.5.3, 3.7.2)
