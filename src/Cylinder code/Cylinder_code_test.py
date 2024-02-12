# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 18:22:12 2021

@author: deepa
"""

import numpy as np

import os

path_codes = 'D:\TAMU\Research\Cylinder code'

os.chdir(path_codes)

from Darboux_cylinder_eqns import *

# Generate random initial position and initial tangent vector

# Pini = np.random.rand(3)
# Tini = np.random.rand(3)
# Tini = Tini/np.linalg.norm(Tini)

# # Generating random vector for tini and setting it to be a unit vector orthogonal to Tini

# tini = np.random.rand(3)
# if abs(np.dot(tini, Tini)) == np.linalg.norm(tini):
    
#     print('Regenerate tini')

# else:
    
#     # Applying Gram-Schmidt
#     temp = tini - np.dot(tini, Tini)*Tini
#     tini = temp/np.linalg.norm(temp)
    
# # Computing uini using cross product between Tini and tini
# uini = np.cross(Tini, tini)

Pini = np.zeros(3)
# Tini = np.array([1, 0, 0])
# tini = np.array([0, 1, 0])
Tini = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
tini = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])
uini = np.array([0, 0, 1])

# Setting other parameters
sini = 0
stotal = 50
s_no_pts = 1000
kg = 0.2
kn = 0.2
taur = 0.3

# s, P1, T, t, u = Darboux_eqns(Tini, tini, uini, Pini, sini, stotal, kg, taur,\
#                              s_no_pts, 0, 199, 'test_positive_kg_positive_taur.pdf')
# s, P2, T, t, u = Darboux_eqns(Tini, tini, uini, Pini, sini, stotal, kg, -taur,\
#                              s_no_pts, 0, 199, 'test_positive_kg_negative_taur.pdf')
# s, P3, T, t, u = Darboux_eqns(Tini, tini, uini, Pini, sini, stotal, -kg, taur,\
#                              s_no_pts, 0, 199, 'test_negative_kg_positive_taur.pdf')
# s, P4, T, t, u = Darboux_eqns(Tini, tini, uini, Pini, sini, stotal, -kg, -taur,\
#                              s_no_pts, 0, 199, 'test_negative_kg_negative_taur.pdf')

s, P1, T, t, u = Darboux_eqns(Tini, tini, uini, Pini, sini, stotal, kg, taur,\
                             s_no_pts, 3, 199, 'test_positive_kg_positive_taur.pdf')
s, P2, T, t, u = Darboux_eqns(Tini, tini, uini, Pini, sini, stotal, kg, -taur,\
                          s_no_pts, 3, 199, 'test_positive_kg_negative_taur.pdf')
 
#%%

plt.rcParams['text.usetex'] = True

# Visualizing all possibilies in a single graph

figure = plt.figure()
ax = Axes3D(figure, elev = 0, azim = 90)
ax.scatter(Pini[0], Pini[1], Pini[2], marker = 'x', linewidth = 1.5,\
           color = 'r', label = 'Initial point')
ax.plot3D(P1[:, 0], P1[:, 1], P1[:, 2], linewidth = 1.5,\
          label = r'Positive $\kappa_g$ and $\tau_r$')
ax.plot3D(P2[:, 0], P2[:, 1], P2[:, 2], linewidth = 1.5, linestyle = '--',\
          color = 'g',label = r'Positive $\kappa_g$ and negative $\tau_r$')
ax.plot3D(P3[:, 0], P3[:, 1], P3[:, 2], linewidth = 1.5, linestyle = '-.',\
          color = 'm',label = r'Negative $\kappa_g$ and positive $\tau_r$')
ax.plot3D(P4[:, 0], P4[:, 1], P4[:, 2], linewidth = 1.5, linestyle = ':',\
          color = 'brown',label = r'Negative $\kappa_g$ and $\tau_r$')
length_vec = 2
ax.plot3D(np.array([Pini[0], Pini[0] + Tini[0]*length_vec]),\
          np.array([Pini[1], Pini[1] + Tini[1]*length_vec]),\
          np.array([Pini[2], Pini[2] + Tini[2]*length_vec]), color = 'black',\
          linewidth = 1.5, label = 'Initial tangent')
ax.plot3D(np.array([Pini[0], Pini[0] + tini[0]*length_vec]),\
          np.array([Pini[1], Pini[1] + tini[1]*length_vec]),\
          np.array([Pini[2], Pini[2] + tini[2]*length_vec]), color = 'orange',\
          linewidth = 1.5, label = 'Initial tangent normal')
ax.plot3D(np.array([Pini[0], Pini[0] + uini[0]*length_vec]),\
          np.array([Pini[1], Pini[1] + uini[1]*length_vec]),\
          np.array([Pini[2], Pini[2] + uini[2]*length_vec]), color = 'purple',\
          linewidth = 1.5, label = 'Initial surface normal')
ax.legend(fontsize = 11, loc = 1, ncol = 2)
ax.set_xlabel('$X$ (m)', fontsize = 14)
ax.set_ylabel('$Y$ (m)', fontsize = 14)
ax.set_zlabel('$Z$ (m)', fontsize = 14)
ax.tick_params(axis = "x", labelsize = 14)
ax.tick_params(axis = "y", labelsize = 14)
ax.tick_params(axis = "z", labelsize = 14)

plt.savefig('test_kg_taur_signs_comparison_XZ_view.pdf', bbox_inches='tight')

plt.show()

#%% Visualizing all possibilities in a single graph with Plotly

import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default='browser'

traces = []

surface = go.Scatter3d(
    x = P1[:, 0], y = P1[:, 1], z = P1[:, 2],
    marker = dict(
        size = 0.2,
        # color = P1[:, 2],
        colorscale = 'Viridis',
    ),
    line = dict(
        color='darkblue'
        # ,
        # width = 0.5
    )
)
traces.append(surface)

surface = go.Scatter3d(
    x = P2[:, 0], y = P2[:, 1], z = P2[:, 2],
    marker = dict(
        size = 0.2,
        # color = P1[:, 2],
        colorscale = 'Viridis',
    ),
    line = dict(
        color='red'
        # ,
        # width = 0.5
    )
)
traces.append(surface)


layout = go.Layout(
    width=800,
    height=700,
    autosize=False,
    scene=dict(
        camera=dict(
            up=dict(
                x=0,
                y=0,
                z=1
            ),
            eye=dict(
                x=0,
                y=1.0707,
                z=1,
            )
        ),
        aspectratio = dict( x=1, y=1, z=0.7 ),
        aspectmode = 'manual'
    ),
)

data = traces
fig = go.Figure(data,layout)

fig.show()
# fig.write_html("file.html")

#%% Curve with the cylinder

Darboux_eqns(Tini, tini, uini, Pini, sini, stotal, kg, taur,\
             s_no_pts, 2, 199, 'test_positive_kg_positive_taur_cylinder.pdf')
Darboux_eqns(Tini, tini, uini, Pini, sini, stotal, kg, -taur,\
             s_no_pts, 2, 199, 'test_positive_kg_negative_taur_cylinder.pdf')
Darboux_eqns(Tini, tini, uini, Pini, sini, stotal, -kg, taur,\
             s_no_pts, 2, 199, 'test_negative_kg_positive_taur_cylinder.pdf')
Darboux_eqns(Tini, tini, uini, Pini, sini, stotal, -kg, -taur,\
             s_no_pts, 2, 199, 'test_negative_kg_negative_taur_cylinder.pdf')
    
#%% Combination of curves

# %matplotlib notebook

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d   

s, P_first, T, t, u = Darboux_eqns(Tini, tini, uini, Pini, sini, stotal, kg, taur,\
                             s_no_pts, 0)

s2, P_second, T2, t2, u2 = Darboux_eqns(T[-1], t[-1], u[-1], P_first[-1], s[-1], s[-1] + stotal, -kg, taur,\
                             s_no_pts, 0)

P_total = np.concatenate((P_first, P_second))

plt.rcParams['text.usetex'] = True

# Visualization

figure = plt.figure()
ax = Axes3D(figure, elev = 20, azim = 50)
ax.scatter(Pini[0], Pini[1], Pini[2], marker = 'x', linewidth = 1.5,\
            color = 'r', label = 'Initial point')
ax.plot3D(P_total[:, 0], P_total[:, 1], P_total[:, 2], linewidth = 1.5,\
          label = r'Concatenation of curves')
length_vec = 2
ax.plot3D(np.array([Pini[0], Pini[0] + Tini[0]*length_vec]),\
          np.array([Pini[1], Pini[1] + Tini[1]*length_vec]),\
          np.array([Pini[2], Pini[2] + Tini[2]*length_vec]), color = 'black',\
          linewidth = 1.5, label = 'Initial tangent')
ax.plot3D(np.array([Pini[0], Pini[0] + tini[0]*length_vec]),\
          np.array([Pini[1], Pini[1] + tini[1]*length_vec]),\
          np.array([Pini[2], Pini[2] + tini[2]*length_vec]), color = 'orange',\
          linewidth = 1.5, label = 'Initial tangent normal')
ax.plot3D(np.array([Pini[0], Pini[0] + uini[0]*length_vec]),\
          np.array([Pini[1], Pini[1] + uini[1]*length_vec]),\
          np.array([Pini[2], Pini[2] + uini[2]*length_vec]), color = 'purple',\
          linewidth = 1.5, label = 'Initial surface normal')
ax.plot3D(np.array([P_first[-1, 0], P_first[-1, 0] + T[-1, 0]*length_vec]),\
          np.array([P_first[-1, 1], P_first[-1, 1] + T[-1, 1]*length_vec]),\
          np.array([P_first[-1, 2], P_first[-1, 2] + T[-1, 2]*length_vec]),\
          linewidth = 1.5, label = 'Tangent at end of first curve')
ax.plot3D(np.array([P_first[-1, 0], P_first[-1, 0] + t[-1, 0]*length_vec]),\
          np.array([P_first[-1, 1], P_first[-1, 1] + t[-1, 1]*length_vec]),\
          np.array([P_first[-1, 2], P_first[-1, 2] + t[-1, 2]*length_vec]),\
          linewidth = 1.5, label = 'Tangent normal at end of first curve')
ax.plot3D(np.array([P_first[-1, 0], P_first[-1, 0] + u[-1, 0]*length_vec]),\
          np.array([P_first[-1, 1], P_first[-1, 1] + u[-1, 1]*length_vec]),\
          np.array([P_first[-1, 2], P_first[-1, 2] + u[-1, 2]*length_vec]),\
          linewidth = 1.5, label = 'Surface normal at end of first curve')
ax.legend(fontsize = 11, loc = 1, ncol = 2)
ax.set_xlabel('$X$ (m)', fontsize = 14)
ax.set_ylabel('$Y$ (m)', fontsize = 14)
ax.set_zlabel('$Z$ (m)', fontsize = 14)
ax.tick_params(axis = "x", labelsize = 14)
ax.tick_params(axis = "y", labelsize = 14)
ax.tick_params(axis = "z", labelsize = 14)

plt.show()

# import plotly.graph_objects as go

# import plotly.io as pio
# pio.renderers.default='browser'

#%% Testing ode function with non-zero kn

kg = 0
kn = 0.2
taur = 0.3

states_initial = np.array([Pini[0], Pini[1], Pini[2], Tini[0], Tini[1],\
                               Tini[2], tini[0], tini[1], tini[2], uini[0],\
                               uini[1], uini[2]])
    
from scipy.integrate import solve_ivp

sol = solve_ivp(ode45_func_Darboux_frame_with_kn, t_span = (sini, sini + stotal),\
                args = (kg, kn, taur), y0 = states_initial, max_step = 0.1)
    
kg = 0.2
taur = 0

sol1 = solve_ivp(ode45_func_Darboux_frame_with_kn, t_span = (sini, sini + stotal),\
                args = (kg, kn, taur), y0 = states_initial, max_step = 0.1)
    
taur = 0.3

sol2 = solve_ivp(ode45_func_Darboux_frame_with_kn, t_span = (sini, sini + stotal),\
                args = (kg, kn, taur), y0 = states_initial, max_step = 0.1)
    
from plotting_class import plotting_functions

fig = plotting_functions()

fig.points_3D([Pini[0]], [Pini[1]], [Pini[2]], 'red', 'Initial point')
fig.scatter_3D(sol.y[0, :], sol.y[1, :], sol.y[2, :], 'black', 'ODE45 Solution kg = 0 taur = kn = 0.2')
fig.scatter_3D(sol1.y[0, :], sol1.y[1, :], sol1.y[2, :], 'purple', 'ODE45 Solution kg = 0.2 taur = 0 kn = 0.2')
fig.scatter_3D(sol2.y[0, :], sol2.y[1, :], sol2.y[2, :], 'pink', 'ODE45 Solution kg = taur = kn = 0.2')
fig.arrows_3D([Pini[0]], [Pini[1]], [Pini[2]], [Tini[0]], [Tini[1]], [Tini[2]],\
              'orange', 'oranges', 'Initial tangent vector', 5, 5, 4, 'n')
fig.arrows_3D([Pini[0]], [Pini[1]], [Pini[2]], [tini[0]], [tini[1]], [tini[2]],\
              'brown', 'Brwnyl', 'Initial tangent normal vector', 5, 5, 4, 'n')
fig.arrows_3D([Pini[0]], [Pini[1]], [Pini[2]], [uini[0]], [uini[1]], [uini[2]],\
              'blue', 'bluyl', 'Initial surface normal vector', 5, 5, 4, 'n')
fig.update_layout_3D('X (m)', 'Y (m)', 'Z (m)', 'Testing variation in kg and taur')
fig.writing_fig_to_html('Trajectory testing.html', 'w')

# figure = plt.figure()
# ax = Axes3D(figure, elev=20, azim=50)
# h1 = ax.scatter(Pini[0], Pini[1], Pini[2], marker = 'x', linewidth = 1.5,\
#            color = 'r')
# ax.set_xlabel('$X$ (m)', fontsize = 14)
# ax.set_ylabel('$Y$ (m)', fontsize = 14)
# ax.set_zlabel('$Z$ (m)', fontsize = 14)
# ax.tick_params(axis = "x", labelsize = 14)
# ax.tick_params(axis = "y", labelsize = 14)
# ax.tick_params(axis = "z", labelsize = 14)

# h3, = ax.plot3D(sol.y[0, :], sol.y[1, :], sol.y[2, :], linewidth = 1.5,\
#           linestyle = '--', label = 'ODE45 solution')

# ax.plot3D(np.array([Pini[0], Pini[0] + Tini[0]]),\
#                   np.array([Pini[1], Pini[1] + Tini[1]]),\
#                   np.array([Pini[2], Pini[2] + Tini[2]]), color = 'red',\
#                   LineWidth = 1.5, label = 'Tangent vector')
# ax.plot3D(np.array([Pini[0], Pini[0] + tini[0]]),\
#           np.array([Pini[1], Pini[1] + tini[1]]),\
#           np.array([Pini[2], Pini[2] + tini[2]]), color = 'brown',\
#           LineWidth = 1.5, label = 'Tangent normal vector')
# ax.plot3D(np.array([Pini[0], Pini[0] + uini[0]]),\
#           np.array([Pini[1], Pini[1] + uini[1]]),\
#           np.array([Pini[2], Pini[2] + uini[2]]), color = 'purple',\
#           LineWidth = 1.5, label = 'Surface normal vector')
# ax.legend(fontsize = 9, loc = 1)
    
#%%

import plotly.graph_objects as go
import pandas as pd
import numpy as np

import plotly.io as pio
pio.renderers.default='browser'

rs = np.random.RandomState()
rs.seed(0)

def brownian_motion(T = 1, N = 100, mu = 0.1, sigma = 0.01, S0 = 20):
    dt = float(T)/N
    t = np.linspace(0, T, N)
    W = rs.standard_normal(size = N)
    W = np.cumsum(W)*np.sqrt(dt) # standard brownian motion
    X = (mu-0.5*sigma**2)*t + sigma*W
    S = S0*np.exp(X) # geometric brownian motion
    return S

dates = pd.date_range('2012-01-01', '2013-02-22')
T = (dates.max()-dates.min()).days / 365
N = dates.size
start_price = 100
y = brownian_motion(T, N, sigma=0.1, S0=start_price)
z = brownian_motion(T, N, sigma=0.1, S0=start_price)

fig = go.Figure(data=go.Scatter3d(
    x=dates, y=y, z=z,
    marker=dict(
        size=4,
        color=z,
        colorscale='Viridis',
    ),
    line=dict(
        color='darkblue',
        width=2
    )
))

fig.update_layout(
    width=800,
    height=700,
    autosize=False,
    scene=dict(
        camera=dict(
            up=dict(
                x=0,
                y=0,
                z=1
            ),
            eye=dict(
                x=0,
                y=1.0707,
                z=1,
            )
        ),
        aspectratio = dict( x=1, y=1, z=0.7 ),
        aspectmode = 'manual'
    ),
)

fig.show()