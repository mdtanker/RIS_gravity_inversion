geo_inversion(
    active_layer='bathymetry',
    exclude_layers=['ice'],
    layers=layers, 
    input_grav=grid_grav, 
    input_grav_column='Gobs_fill',
    regional_method='filter',
    filter='g200e3', 
    trend_order=1,
    deriv_type='annulus',
    reset=True,
    constraints=False,
    Max_Iterations=2,
    max_layer_change_per_iter=100,
    misfit_sq_tolerance=0.00001,
    delta_misfit_squared_tolerance=0.002,
    plot=True,
    plot_type='xarray'
    ) 
# 15 mins 200km zoom/5k, 7mins 400kmzoom/5k (113x113 prisms, 33x33 grav)
# 1:54s for annulus
# 1:46s for prisms
# notify
