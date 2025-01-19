# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np


# class PauseAnimation:
#     def __init__(self):
#         fig, ax = plt.subplots()
#         ax.set_title('Click to pause/resume the animation')
#         x = np.linspace(-0.1, 0.1, 1000)

#         # Start with a normal distribution
#         self.n0 = (1.0 / ((4 * np.pi * 2e-4 * 0.1) ** 0.5)
#                    * np.exp(-x ** 2 / (4 * 2e-4 * 0.1)))
#         self.p, = ax.plot(x, self.n0)

#         self.animation = animation.FuncAnimation(
#             fig, self.update, frames=200, interval=50, blit=True)
#         self.paused = False

#         fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

#     def toggle_pause(self, *args, **kwargs):
#         if self.paused:
#             self.animation.resume()
#         else:
#             self.animation.pause()
#         self.paused = not self.paused

#     def update(self, i):
#         self.n0 += i / 100 % 5
#         self.p.set_ydata(self.n0 % 20)
#         return (self.p,)


# pa = PauseAnimation()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.widgets as widgets

# Create some sample data
t = np.arange(0, 20, 0.1)
x = np.sin(t)
y = np.cos(t)
z = t

# Create the figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set up the initial plot
line, = ax.plot(x[:1], y[:1], z[:1])

# Animation update function
def update(i):
    line.set_data(x[:i], y[:i])
    line.set_3d_properties(z[:i])
    return line, 

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)

# Create play/pause button
play_pause_ax = fig.add_axes([0.85, 0.05, 0.1, 0.075])
play_pause_button = widgets.Button(play_pause_ax, 'Pause')
is_paused = False

def play_pause(event):
    global is_paused
    if is_paused:
        ani.event_source.start()
        play_pause_button.label.set_text('Pause')
    else:
        ani.event_source.stop()
        play_pause_button.label.set_text('Play')
    is_paused = not is_paused

play_pause_button.on_clicked(play_pause)

plt.show()