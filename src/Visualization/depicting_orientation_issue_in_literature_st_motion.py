from view_manager import ViewManager
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import time
from msg_state import MsgState
import math
import numpy as np
from math import sin as sin
from math import cos as cos
from math import sqrt as sqrt
from rotations import euler_to_rotation

plt.rcParams['text.usetex'] = True

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

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.tight_layout()
plt.ion()
viewers = ViewManager(ax, animation=True, video=True, scale_aircraft=4.0, video_name = 'st_motion_heading_45_deg_pitch_0_deg_roll_0_deg.mp4')
# viewers = ViewManager(ax, animation=True, video=True, scale_aircraft=4.0, video_name = 'st_motion_heading_45_deg_pitch_0_deg_roll_45_deg.mp4')

# Constructing an array for the vehicle to track
pos_global = [[i/2, i/2, 10] for i in range(100)]
tang_global_path = [[1/math.sqrt(2), 1/math.sqrt(2), 0] for i in range(100)]
# tang_normal_path = [[0, 0, 1] for i in range(50)]
tang_normal_path = [[-1/math.sqrt(2), 1/math.sqrt(2), 0] for i in range(100)]
# tang_normal_path = [[math.cos(45*math.pi/180)*(-1/math.sqrt(2)), math.cos(45*math.pi/180)*1/math.sqrt(2), math.cos(45*math.pi/180)*1] for i in range(100)]
surf_normal_path = [np.cross(tang_global_path[i], tang_normal_path[i]) for i in range(100)]

# pos_global = [[20, 20, 20] for i in range(50)]

# roll_angle = [30*math.pi/180 for i in range(50)]
# pitch_angle = [6*i*math.pi/180 for i in range(50)]
# yaw_angle = [90*math.pi/180 for i in range(50)]

# tang_global_path = [[0, 1, 0] for i in range(50)]
# # tang_normal_path = [np.dot(np.array([[1, 0, 0], [0, math.cos(i/20), -math.sin(i/20)], [0, math.sin(i/20), math.cos(i/20)]]), [0, 0, 1]) for i in range(50)]
# tang_normal_path = [[0, 0, 1] for i in range(50)]
# surf_normal_path = [np.cross(tang_global_path[i], tang_normal_path[i]) for i in range(50)]

# tang_global_path = []; tang_normal_path = []; surf_normal_path = []
# for i in range(len(roll_angle)):

#     alpha = yaw_angle[i]; beta = pitch_angle[i]; gamma = roll_angle[i]
#     Rz = np.array([
#         [np.cos(alpha), -np.sin(alpha), 0],
#         [np.sin(alpha), np.cos(alpha), 0],
#         [0, 0, 1]
#     ])
    
#     # Rotation around Y-axis (Pitch)
#     Ry = np.array([
#         [np.cos(beta), 0, np.sin(beta)],
#         [0, 1, 0],
#         [-np.sin(beta), 0, np.cos(beta)]
#     ])
    
#     # Rotation around X-axis (Roll)
#     Rx = np.array([
#         [1, 0, 0],
#         [0, np.cos(gamma), -np.sin(gamma)],
#         [0, np.sin(gamma), np.cos(gamma)]
#     ])
    
#     # Final combined rotation matrix (ZYX)
#     R_zyx = np.dot(Rz, np.dot(Ry, Rx))

#     tang_global_path.append([R_zyx[0, 0], R_zyx[1, 0], R_zyx[2, 0]])
#     tang_normal_path.append([R_zyx[0, 1], R_zyx[1, 1], R_zyx[2, 1]])
#     surf_normal_path.append([R_zyx[0, 2], R_zyx[1, 2], R_zyx[2, 2]])

true_state = MsgState()

print([pos_global[i][0] for i in range(len(pos_global))])

ax.plot3D([pos_global[i][0] for i in range(len(pos_global))],\
          [pos_global[i][1] for i in range(len(pos_global))],\
          [pos_global[i][2] for i in range(len(pos_global))], linewidth = 1.5, label = 'Trajectory')

ax.view_init(elev = 30, azim = -130)

ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.set_zlim(0, 50)

ax.set_xlabel(r'$X$ (m)', fontsize = 24, labelpad = 10)
ax.set_ylabel(r'$Y$ (m)', fontsize = 24, labelpad = 10)
ax.set_zlabel(r'$Z$ (m)', fontsize = 24, labelpad = 10)

ax.tick_params(axis='both', which='major', labelsize = 18)
ax.tick_params(axis='both', which='minor', labelsize = 18)

# We define the length of the arrow for representing the orientation
# length = 10
length = 15
locplot = None; tangplot = None; surfplot = None; tangnorm_plot = None

for i in range(len(pos_global)):

    # If a plot already exists, we remove it
    if locplot:
        locplot.remove()
        tangplot.remove()
        tangnorm_plot.remove()
        surfplot.remove()
        
    # We plot the current configuration
    locplot = ax.scatter(pos_global[i][0], pos_global[i][1], pos_global[i][2], marker = 'o', linewidth = 1.5,\
        color = 'k')
    tangplot = ax.arrow3D(pos_global[i][0], pos_global[i][1], pos_global[i][2], length*tang_global_path[i][0],\
            length*tang_global_path[i][1], length*tang_global_path[i][2], mutation_scale=20, fc='red', label = 'Tangent vector')
    tangnorm_plot = ax.arrow3D(pos_global[i][0], pos_global[i][1], pos_global[i][2], length*tang_normal_path[i][0],\
                length*tang_normal_path[i][1], length*tang_normal_path[i][2], mutation_scale=20, fc='blue', label = 'Tangent normal vector')
    surfplot = ax.arrow3D(pos_global[i][0], pos_global[i][1], pos_global[i][2], length*surf_normal_path[i][0],\
                length*surf_normal_path[i][1], length*surf_normal_path[i][2], mutation_scale=20, fc='green', label = 'Surface normal vector')
    
    # ax.legend(fontsize = 9, loc = 1)
    ax.legend(fontsize = 16, loc = 9, ncol = 2)

    # We update the state of the aircraft
    true_state.north = pos_global[i][0]
    true_state.east = pos_global[i][1]
    true_state.altitude = -pos_global[i][2]

    # # We plot the orientation of the vehicle as well
    # true_state.r11 = tang_global_path[i][0]
    # true_state.r21 = tang_global_path[i][1]
    # true_state.r31 = tang_global_path[i][2]
    # true_state.r12 = tang_normal_path[i][0]
    # true_state.r22 = tang_normal_path[i][1]
    # true_state.r23 = tang_normal_path[i][2]
    # true_state.r13 = surf_normal_path[i][0]
    # true_state.r23 = surf_normal_path[i][1]
    # true_state.r33 = surf_normal_path[i][2]

    # Calculating the angles. The net rotation matrix is given below. Here, psi is yaw angle, theta is pitch, and phi is roll angle.
    """cψcθ −sψcφ + cψsθsφ sψsφ + cψsθcφ
    sψcθ cψcφ + sψsθsφ −cψsφ + sψsθcφ
    −sθ cθsφ cθcφ"""

    if math.sqrt((tang_global_path[i][0])**2 + (tang_global_path[i][1])**2) <= 10**(-8):
        
        pitch_angle = [-np.sign(tang_global_path[i][2])*math.pi/2]
        # We set roll angle to be zero
        roll_angle = 0.0
        yaw_angle = math.atan2(-tang_normal_path[i][0], tang_normal_path[i][1])

    else:

        pitch_angle = math.atan2(-tang_global_path[i][2], math.sqrt((tang_global_path[i][0])**2 + (tang_global_path[i][1])**2))
        yaw_angle = math.atan2(tang_global_path[i][1], tang_global_path[i][0])
        roll_angle = math.atan2(tang_normal_path[i][2], surf_normal_path[i][2])

    # true_state.psi = yaw_angle[i]
    # true_state.theta = pitch_angle[i]
    # true_state.phi = roll_angle[i]
    true_state.psi = yaw_angle
    true_state.theta = pitch_angle
    true_state.phi = roll_angle

    # print('True state is for i = ', i, ' is ', true_state.north, true_state.east, true_state.altitude)

    # Plotting the aircraft
    viewers.update(
        fig,
        time.time(),
        true_state = true_state,  # true states
    )

    # if i == 0:
    #     # Saving the figure
    #     fig.savefig('roll_angle_45_deg.png', dpi = 300)

    # print('Printing the ', i, 'th point')
    plt.pause(0.1)

# We close the simulation
viewers.close()

plt.ioff()  # Turn off interactive mode

# We show the trajectory plot
plt.show()

print('Done.')