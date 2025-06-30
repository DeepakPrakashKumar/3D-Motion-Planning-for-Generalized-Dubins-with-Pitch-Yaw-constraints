module BasicTest
using Dubins3D
using DelimitedFiles

# Convert degreees to radians
deg2rad(x) = pi * x / 180.

# Initial and final configurations [x, y, z, heading angle, pitch angle]
# qi = [200., 500., 200., deg2rad(180.), deg2rad(-5.)]
# qf = [500., 350., 100., deg2rad(0.), deg2rad(-5.)]

LONG = [
  [
    [200, 500, 200, deg2rad(180), deg2rad(-5)],
    [500, 350, 100, deg2rad(0), deg2rad(-5)],
    [467.70, 449.56],
    "Long 1"
  ],[
    [100, -400, 100, deg2rad(30), deg2rad(0)],
    [500, -700, 0, deg2rad(150), deg2rad(0)],
    [649.52, 636.12],
    "Long 2"
  ],[
    [-200, 200, 250, deg2rad(240), deg2rad(15)],
    [ 500, 800,   0, deg2rad(45), deg2rad(15)],
    [1088.10, 1063.41],
    "Long 3"
  ],[
    [-300, 1200, 350, deg2rad(160), deg2rad(0)],
    [1000,  200,   0, deg2rad(30), deg2rad(0)],
    [1802.60, 1789.21],
    "Long 4"
  ],[
    [-500, -300, 600, deg2rad(150), deg2rad(10)],
    [1200,  900, 100, deg2rad(300), deg2rad(10)],
    [2245.14, 2216.40],
    "Long 5"
  ]
]

SHORT = [
  [
    [120, -30, 250, deg2rad(100), deg2rad(-10)],
    [220, 150, 100, deg2rad(300), deg2rad(-10)],
    [588.60, 583.47],
    "Short 1"
  ],[
    [380, 230, 200, deg2rad(30), deg2rad(0)],
    [280, 150,  30, deg2rad(200), deg2rad(0)],
    [667.71, 658.53],
    "Short 2"
  ],[
    [-80, 10, 250, deg2rad(20), deg2rad(0)],
    [ 50, 70,   0, deg2rad(240), deg2rad(0)],
    [979.34, 968.25],
    "Short 3"
  ],[
    [400, -250, 600, deg2rad(350), deg2rad(0)],
    [600, -150, 300, deg2rad(150), deg2rad(0)],
    [1169.73, 1161.55],
    "Short 4"
  ],[
    [-200, -200, 450, deg2rad(340), deg2rad(0)],
    [-300,  -80, 100, deg2rad(100), deg2rad(0)],
    [1367.56, 1354.12],
    "Short 5"
  ]
]

ADDITIONAL = [
    [
        [120, 40, 20, deg2rad(90), deg2rad(-5)],
        [300, 40, 15, deg2rad(-90), deg2rad(-5)],
        [],
        "Additional 1"
    ],
    [
        [120, 40, 20, deg2rad(90), deg2rad(-15)],
        [130, 120, 41, deg2rad(85), deg2rad(20)],
        [],
        "Additional 2"
    ]
]

# Minimum turning radius
rhomin = 40.
# Pich angle constraints [min_pitch, max_pitch]
pitchmax = deg2rad.([-15., 20.])

# for data in [LONG, SHORT]
# data = LONG
data = ADDITIONAL
for i in 1:length(data)

    print("NAME ", data[i][4] , " -------------  ")
    d = data[i]
    print(d[1], " ", d[2], " ", d[3], " ", d[4], "\n")
    qi = d[1]
    qf = d[2]
    dubins = DubinsManeuver3D(qi, qf, rhomin, pitchmax)
    println("Length: ", dubins.length)

    # maneuver = DubinsManeuver3D(qi, qf, rhomin, pitchmax)

    # # Length of the 3D Dubins path
    # @show maneuver.length

    # # Sample the manever by 500 samples
    # samples = compute_sampling(maneuver; numberOfSamples = 500)
    # # First and last samples - should be equal to qi and qf
    # @show samples[1]
    # @show samples[end]

    # @show samples

    # # We now plot the path
    # for i in 1:length(maneuver.path)
    #     plot(maneuver.path[i].maneuver)
    # end

    # using Plots

    # # We plot the initial and final configurations

    # scale = 10.
    # p = plot3d([qi[1]], [qi[2]], [qi[3]], seriestype = :quiver,
    #        quiver=([scale*cos(qi[4])*cos(qi[5])], [scale*sin(qi[4])*cos(qi[5])], [scale*sin(qi[5])]),
    #        label = "Arrow", arrow = true)

    # print(qi[1], qi[2], qi[3], qi[4], qi[5])

    # p = plot3d(samples[:][1], samples[:][2], samples[:][3], seriestype=:line, marker=:circle, label="Maneuver")

    # display(p)
    # using DelimitedFiles

    # writedlm("samples.csv", samples, ',')

    # @show size(samples)
    # @show eltype(samples)

    println("Constructing matrix")
    # @show dubins
    samples = compute_sampling(dubins; numberOfSamples = 500)

    # Turn your samples into a 2D matrix first
    samples_matrix = hcat(samples...)'  # now size(samples_matrix) == (500, 5)

    println("Writing matrix")
    @show size(samples_matrix)
    @show eltype(samples_matrix)   

    # We provide the column names
    header = ["X" "Y" "Z" "Heading" "Pitch"]
    println(header)
    open(d[4] * ".csv", "w") do io
        writedlm(io, header, ',')
        writedlm(io, samples_matrix, ',')
    end
end
# end
end