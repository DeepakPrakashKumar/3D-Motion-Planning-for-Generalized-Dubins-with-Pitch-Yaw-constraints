# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation
# # import numpy as np


# # class PauseAnimation:
# #     def __init__(self):
# #         fig, ax = plt.subplots()
# #         ax.set_title('Click to pause/resume the animation')
# #         x = np.linspace(-0.1, 0.1, 1000)

# #         # Start with a normal distribution
# #         self.n0 = (1.0 / ((4 * np.pi * 2e-4 * 0.1) ** 0.5)
# #                    * np.exp(-x ** 2 / (4 * 2e-4 * 0.1)))
# #         self.p, = ax.plot(x, self.n0)

# #         self.animation = animation.FuncAnimation(
# #             fig, self.update, frames=200, interval=50, blit=True)
# #         self.paused = False

# #         fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

# #     def toggle_pause(self, *args, **kwargs):
# #         if self.paused:
# #             self.animation.resume()
# #         else:
# #             self.animation.pause()
# #         self.paused = not self.paused

# #     def update(self, i):
# #         self.n0 += i / 100 % 5
# #         self.p.set_ydata(self.n0 % 20)
# #         return (self.p,)


# # pa = PauseAnimation()
# # plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
# import matplotlib.widgets as widgets

# # Create some sample data
# t = np.arange(0, 20, 0.1)
# x = np.sin(t)
# y = np.cos(t)
# z = t

# # Create the figure and axes
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Set up the initial plot
# line, = ax.plot(x[:1], y[:1], z[:1])

# # Animation update function
# def update(i):
#     line.set_data(x[:i], y[:i])
#     line.set_3d_properties(z[:i])
#     return line, 

# # Create the animation
# ani = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)

# # Create play/pause button
# play_pause_ax = fig.add_axes([0.85, 0.05, 0.1, 0.075])
# play_pause_button = widgets.Button(play_pause_ax, 'Pause')
# is_paused = False

# def play_pause(event):
#     global is_paused
#     if is_paused:
#         ani.event_source.start()
#         play_pause_button.label.set_text('Pause')
#     else:
#         ani.event_source.stop()
#         play_pause_button.label.set_text('Play')
#     is_paused = not is_paused

# play_pause_button.on_clicked(play_pause)

# plt.show()

from view_manager import ViewManager
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import time
from msg_state import MsgState
import math
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
viewers = ViewManager(ax, animation=True, video=False, video_name = 'trajectory_aircraft.mp4')

# Constructing an array for the vehicle to track
pos_global = [[i, i, i] for i in range(50)]
tang_global_path = [[1/math.sqrt(3), 1//math.sqrt(3), 1/math.sqrt(3)] for i in range(50)]
tang_normal_path = [[-1/math.sqrt(2), 1/math.sqrt(2), 0] for i in range(50)]
surf_normal_path = [np.cross(tang_global_path[i], tang_normal_path[i]) for i in range(50)]
true_state = MsgState()

print([pos_global[i][0] for i in range(len(pos_global))])

ax.plot3D([pos_global[i][0] for i in range(len(pos_global))],\
          [pos_global[i][1] for i in range(len(pos_global))],\
          [pos_global[i][2] for i in range(len(pos_global))], linewidth = 1.5, label = 'Trajectory')

for i in range(len(pos_global)):

    # We update the state of the aircraft
    true_state.north = pos_global[i][0]
    true_state.east = pos_global[i][1]
    true_state.altitude = -pos_global[i][2]

    # We plot the orientation of the vehicle as well
    true_state.r11 = tang_global_path[i][0]
    true_state.r21 = tang_global_path[i][1]
    true_state.r31 = tang_global_path[i][2]
    true_state.r12 = tang_normal_path[i][0]
    true_state.r22 = tang_normal_path[i][1]
    true_state.r23 = tang_normal_path[i][2]
    true_state.r13 = surf_normal_path[i][0]
    true_state.r23 = surf_normal_path[i][1]
    true_state.r33 = surf_normal_path[i][2]

    # print('True state is for i = ', i, ' is ', true_state.north, true_state.east, true_state.altitude)

    # Plotting the aircraft
    viewers.update(
        time.time(),
        true_state = true_state,  # true states
    )

    # print('Printing the ', i, 'th point')
    plt.pause(.05)

# We show the trajectory plot
plt.show()