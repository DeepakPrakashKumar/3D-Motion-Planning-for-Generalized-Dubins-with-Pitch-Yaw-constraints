def Path_generation_sphere_sphere_sphere(ini_config, fin_config, center_ini_sphere, center_fin_sphere,\
                                           r, R, axis_plane, ht_plane, disc_no, plot_figure_configs,\
                                           visualization = 1, filename = "temp.html"):
    '''
    In this function, the paths connecting a given pair of spheres (inner or outer) with
    an interemediary sphere is generated.

    Parameters
    ----------
    ini_config : Numpy 4x3 array
        Contains the initial position in the first row, the direction cosines of
        the initial tangent vector in the second row, the direction cosines of the
        initial tangent normal vector in the third row, and the direction cosines
        of the surface normal vector in the fourth row.
    fin_config : Numpy 4x3 array
        Contains the final position vector, the direction cosines of the final
        tangent vector, the tangent normal vector, and the surface normal vector
        in the same format as the ini_config variable.
    center_ini_sphere : Array
        Contains the position of the center of the initial sphere.
    center_fin_sphere : Array
        Contains the position of the center of the final sphere.
    r: Scalar
        Radius of the tight turn.
    R : Scalar
        Radius of the surface.
    axis_plane : Array
        Axis of the line connecting the centers of the two spheres.
    ht_plane : Scalar
        Length of the plane connecting the considered pair of spheres.
    disc_no : Scalar
        Number of discretizations in the parameters considered, which correspond
        to the angle corresponding to parameterizing the center of the intermediary sphere,
        the angle corresponding to the tangent vector corresponding to exit from the initial
        sphere, and the angle corresponding to the tangent vector for entry onto the final sphere.
    plot_figure_configs : Plotly figure handle
        Figure handle corresponding to the plotly figure generated, which is utilized and
        updated if visualization = 1.
    visualization : Scalar, optional
        Variable to decide whether to show the plot of the configurations and the
        surfaces. Default is equal to 1.
    filename : String, optional
        Name of the file in which the figure should be written. Used when visualization
        is set to 1. Default is "temp.html".

    Returns
    -------

    '''

    # Discretizing the angle for parameterizing the intermediary sphere and the
    # angles for the tangent vector for exit from initial sphere and entry into final
    # sphere
    theta