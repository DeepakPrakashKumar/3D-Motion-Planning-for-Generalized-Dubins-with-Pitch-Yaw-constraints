using Plots

# Example qi = [x, y, z, yaw, pitch]
qi = [0.0, 0.0, 0.0, 0.0, 0.0]  # All zeros

# Set a scale so the arrow is visible
scale = 2.0

# Compute direction
dx = scale * cos(qi[4]) * cos(qi[5])
dy = scale * sin(qi[4]) * cos(qi[5])
dz = scale * sin(qi[5])

# Now plot
p = plot3d([qi[1]], [qi[2]], [qi[3]],
    seriestype = :quiver,
    quiver = ([dx], [dy], [dz]),
    arrow = true,
    label = "Arrow",
    xlims = (-3, 3),
    ylims = (-3, 3),
    zlims = (-3, 3))

display(p)
