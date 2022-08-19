import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse.linalg import lsqr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pygmt
from pyproj import Transformer
import pyvista as pv
import geopandas as gpd
import harmonica as hm
import verde as vd
import xarray as xr
from tqdm import tqdm
import rioxarray
import seaborn as sns
import antarctic_plots.utils as utils
import warnings
from typing import Union


warnings.filterwarnings('ignore', message="pandas.Int64Index")
warnings.filterwarnings('ignore', message="pandas.Float64Index")

def import_layers(
            layers_list, 
            spacing_list, 
            rho_list, 
            fname_list,
            grav_spacing, 
            active_layer,
            buffer_region,
            inversion_region,
            grav_file,
            constraints=False,
            plot=True,
            **kwargs):
    """
    Import layers, files, and their properties to prep for gravity inversion.

    Parameters
    ----------
    layers_list : list
        _description_
    spacing_list : list
        _description_
    rho_list : list
        _description_
    fname_list : list
        grids should be netcdf, tiff's might work. Should be pixel registered.
    grav_spacing : int
        _description_
    active_layer : str
        _description_
    buffer_region : np.ndarray
        _description_
    inversion_region : np.ndarray
        _description_
    grav_file : str
        _description_
    constraints : bool, optional
        Choose whether to process constraints grid and df, by default False
    plot : bool, optional
        choose whether to plot layers or not, by default True
    
    Other Parameters
    ----------------
    **kwargs: dict
        plot_region: np.ndarray or str
            region to plot for 2d or 3d plots, by default is inversion_region
        constraints_file: str
            .nc file of a constraints grid, 0-1.
         constraints_points: str
            .csv file with position of constraints
        plot_type: str
            choose method of plotting; either "xarray" or "pygmt", by default is "xarray"
        power_spectrum: bool
            choose to plot radially average power spectrum of layers, by defaul is False
    Returns
    -------
    tuple
        layers (dict), 
        grav (pd.DataFrame), 
        constraints_df (pd.DataFrame), 
        constraints_RIS_df (pd.DataFrame)
    """
    layers_list = pd.Series(layers_list)
    spacing_list = pd.Series(spacing_list)
    rho_list = pd.Series(rho_list)
    fname_list = pd.Series(fname_list)

    if grav_file[-3:] == '.nc':
        # read and resample gravity grid
        filt_grav = pygmt.grdfilter(
                            grid=grav_file,
                            filter=f'g{grav_spacing}', 
                            distance='0',
                            # nans='r', # retains NaNs
                            registration='p',
                            verbose='q')                 
        grav = pygmt.grdsample(
                            grid=filt_grav,
                            region=inversion_region, 
                            registration='p', 
                            spacing=grav_spacing,
                            verbose='q').to_dataset(name='Gobs')
        # center on 0
        grav['Gobs'] -= grav.Gobs.mean().item()
        """
        below line makes it so grav is calculated in entire grid domain
        if dont want that, set to:
        grav['z'] = (grav.Gobs*0) + 1e3    
        """
        grav['z'] = xr.ones_like(grav.Gobs) + 1e3 
        grav.to_netcdf('tmp_outputs/tmp_grav.nc')

    if grav_file[-3:] == 'csv':
        # read file
        df=pd.read_csv(grav_file, index_col=False)
        # block reduce
        grav = pygmt.blockmedian(df[['x','y','Gobs']], spacing=grav_spacing, 
                region=inversion_region, registration='p')
        grav['z'] = pygmt.blockmedian(df[['x','y','z']], spacing=grav_spacing, 
                region=inversion_region, registration='p').z
        # with Verde
        # reducer_mean = vd.BlockReduce(reduction=np.mean, spacing=grav_spacing/2, 
        #     center_coordinates=False)
        # coordinates, data = reducer_mean.filter(
        #     coordinates=(df.x, df.y), data=(df.Gobs, df.z))
        # blocked = pd.DataFrame(data={'x': coordinates[0], 'y':coordinates[1], 
        #     'Gobs':data[0], 'z':data[1]})
        # grav = blocked[vd.inside((blocked.x, blocked.y), inversion_region)].copy()
        grav['Gobs'] -= grav.Gobs.mean()

    # make nested dictionary for layers and properties
    layers = {j:{'spacing':spacing_list[i], 
            'fname':fname_list[i], 
            'rho':rho_list[i]} for i, j in enumerate(layers_list)}

    # read and resample layer grids, convert to dataframes
    for k, v in layers.items():
        if int(v['spacing']) > int(pygmt.grdinfo(v['fname'], 
            per_column=True, o=7)[:-1]):
            print(
                f"filtering and resampling {k} from", 
                f"{int(pygmt.grdinfo(v['fname'], per_column=True, o=7)[:-1])}m",
                f"to {int(v['spacing'])}m")
            v['grid'] = pygmt.grdfilter(
                        grid=v['fname'],
                        region=buffer_region, 
                        registration='p', 
                        spacing=v['spacing'],
                        filter=f"g{v['spacing']}", 
                        distance='0',
                        verbose='q') 
            if constraints==True:
                # read and resample constraints grid, and mask outside of RIS
                constraints_grid = pygmt.grdfilter(
                        grid=kwargs.get('constraints_file'),
                        region=buffer_region, 
                        registration='p', 
                        spacing=spacing_list[layers_list[layers_list == active_layer].index[0]],
                        filter=f"g{spacing_list[layers_list[layers_list == active_layer].index[0]]}", 
                        distance='0',
                        verbose='q') 
        else:
            print(
                f"resampling {k} from", 
                f"{int(pygmt.grdinfo(v['fname'], per_column=True, o=7)[:-1])}m",
                f"to {int(v['spacing'])}m")
            v['grid'] = pygmt.grdsample(
                        grid=v['fname'],
                        region=buffer_region, 
                        registration='p', 
                        spacing=v['spacing'],
                        verbose='q')
            if constraints==True:
                # read and resample constraints grid, and mask outside of RIS
                constraints_grid = pygmt.grdsample(
                        grid=kwargs.get('constraints_file'),
                        region=buffer_region, 
                        registration='p', 
                        spacing=spacing_list[layers_list[layers_list == active_layer].index[0]],
                        verbose='q') 

        v['df'] = v['grid'].to_dataframe().reset_index()
        v['df']['rho']=v['rho']
        v['df'].dropna(how='any', inplace=True)
        v['len']=len(v['df'].x) 

    if constraints==True:
        constraints_df =  pd.read_csv(kwargs.get("constraints_points"), index_col=False)
        mask = utils.mask_from_shp("plotting/MEaSUREs_RIS.shp", masked=True, 
            invert=False, region=buffer_region, spacing=1e3,)
        mask.to_netcdf('tmp_outputs/tmp_mask.nc')
        constraints_RIS_df = pygmt.select(data=constraints_df, gridmask='tmp_outputs/tmp_mask.nc',)

    # print lengths
    for k, v in layers.items():
        print(f"{k}: {v['len']} points, elevations:"
              f"{int(np.nanmax(v['grid']))}m to {int(np.nanmin(v['grid']))}m") 
    
    if grav_file[-3:] == 'csv':
        print(f'gravity: {len(grav)} points')  
        print(f'gravity avg. elevation: {int(grav.z.max())}')
    elif grav_file[-3:] == '.nc':
        print(f'gravity: {len(vd.grid_to_table(grav.Gobs))} points')
        print(f'gravity avg. elevation: {int(np.nanmax(grav.z))}')    

    if constraints==True:
        print(f'bathymetry control points:{len(constraints_df)}') 

    if plot==True:
        if kwargs.get("plot_type", 'xarray') =='pygmt':
            utils.plot_grd(
                grid = grid_grav.Gobs, 
                plot_region=kwargs.get('plot_region', inversion_region), 
                cmap = "jet",
                grd2cpt_name = 'grav',
                cbar_label = "observed gravity (mGal)", 
                constraints = constraints,    
                )

            for i, (k, v) in enumerate(layers.items()):
                utils.plot_grd(
                    grid=layers[k]['grid'], 
                    plot_region=kwargs.get('plot_region', inversion_region), 
                    # cmap = "plotting/layer.cpt",
                    cmap='viridis',
                    grd2cpt_name = 'elevations',
                    cbar_label = f"{k} elevation (m)",
                    origin_shift='xshift',
                    )
            fig.show(width=1200)
        elif kwargs.get("plot_type", 'xarray') == 'xarray':
            if constraints == True:
                extra=2
            else:
                extra=1
            fig, ax = plt.subplots(ncols=len(layers)+extra, nrows=1, figsize=(20,20))
            p=0
            if grav_file[-3:] == 'csv':
                grav_grid = pygmt.xyz2grd(data=grav, 
                        region=inversion_region, 
                        spacing=grav_spacing,  
                        registration='p')
            elif grav_file[-3:] == '.nc':
                grav_grid = grav.Gobs
            grav_grid.plot(ax=ax[p], robust=True, cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax[p].set_title('Observed gravity')
            p+=1
            if constraints == True:
                constr = constraints_grid.rio.set_spatial_dims(
                                    'x', 'y').rio.write_crs("epsg:3031").rio.clip_box(
                                            minx=inversion_region[0], 
                                            maxx=inversion_region[1], 
                                            miny=inversion_region[2], 
                                            maxy=inversion_region[3])
                constr.plot(ax=ax[p], robust=True, cmap='copper', 
                    cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
                ax[p].set_title('Constraints grid')
            for i, j in enumerate(layers):    
                grid = layers[j]['grid'].rio.set_spatial_dims(
                                'x', 'y').rio.write_crs("epsg:3031").rio.clip_box(
                                    minx=kwargs.get('plot_region', inversion_region)[0], 
                                    maxx=kwargs.get('plot_region', inversion_region)[1], 
                                    miny=kwargs.get('plot_region', inversion_region)[2], 
                                    maxy=kwargs.get('plot_region', inversion_region)[3])
                percent = 1
                lims = (np.nanquantile(grid, q=percent/100),
                    np.nanquantile(grid, q=1-(percent/100)))
                grid.plot(ax=ax[i+extra], vmin=lims[0], vmax=lims[1], cmap='gist_earth',
                    cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})
                ax[i+extra].set_title(f'{j} elevation')
                if constraints == True:
                    if j == active_layer:
                        ax[i+extra].plot(constraints_RIS_df.x, constraints_RIS_df.y, 
                        'r+')
                ax[i+extra].add_patch(mpl.patches.Rectangle(
                    xy = (inversion_region[0], inversion_region[2]), 
                    width = (inversion_region[1]-inversion_region[0]),
                    height = (inversion_region[3]-inversion_region[2]),
                    linewidth=1, fill=False))
            for a in ax:
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_xlabel('')
                a.set_ylabel('')
                a.set_aspect('equal')
    if kwargs.get('power_spectrum', False) is True:
        names = []
        grids = []
        for k, v in layers.items():
            names.append(k)
            grids.append(v['grid'])
        names.append('observed_gravity')
        grids.append(grav_grid)
        utils.raps(grids, names)

    if constraints is True:
        return (layers, grav.astype(np.float64), grav_spacing, constraints_grid, constraints_df, constraints_RIS_df) 
    else: 
        return (layers, grav.astype(np.float64), grav_spacing, None, None)

def grids_to_prism_layers(
    layers: dict, 
    buffer_region: Union[str, np.ndarray] = None,
    plot_region: Union[str, np.ndarray] = None, 
    plot: bool = False, 
    **kwargs
    ):
    """
    Turn nested dictionary of grids into series of vertical prisms between each layer 
    and plot in either 3D or as 2D as prism thickness.

    Parameters
    ----------
    layers : dict
        Nested dict; where each layer is a dict with keys: 
            'spacing': int, float; grid spacing 
            'fname': str; grid file name
            'rho': int, float; constant density value for layer
            'grid': xarray.DataArray;  
            'df': pandas.DataFrame; 2d representation of grid
    buffer_region : Union[str, np.ndarray], optional
        region including buffer zone, by default reads region from first grid layer
    plot_region : Union[str, np.ndarray], optional
        GMT-format region to plot for both 2d and 3d plots, by default is buffer region
    plot : bool, optional
        choose whether to plot results, by default False
    
    Other Parameters
    ----------------
    **kwargs : dict
        layers_for_3d: list 
            list of layers to include in 3D plot, by default is all layers.
        clip_cube: bool
            choose to clip out cube of 3D plot, by default is True.
        plot_type : str, optional
            choose between plotting prism layers in '2D' or '3D', by default '3D'.
        inversion_region: : Union[str, np.ndarray], optional
            inner inversion region, used to plot polygon
    """
    plot_type = kwargs.get("plot_type", "3D")
    inversion_region = kwargs.get("inversion_region", None)
    # buffer region defaults to first layer extent
    if buffer_region is None:
        buffer_region = [int(pygmt.grdinfo(
            list(layers.values())[0]['grid'], 
            per_column="n", o=i)[:-1]) for i in range(4)]
    
    # plot region defaults to buffer region
    if plot_region is None:
        plot_region = buffer_region
    
    # add density variable to datasets
    for k, v in layers.items():
        v['grid']['density']=v['grid'].copy()
        v['grid'].density.values[:] = v['rho']

    # list of layers, bottom up
    # reversed_layers_list = layers_list.iloc[::-1]
    reversed_layers_list = pd.Series([k for k,v in layers.items()]).iloc[::-1]

    # create prisms layers from input grids
    for i, j in enumerate(reversed_layers_list):
        if i == 0:
            layers[j]['prisms']=hm.prism_layer(
                coordinates=(layers[j]['grid'].x.values, layers[j]['grid'].y.values),    
                surface=layers[j]['grid'], 
                # reference=-50e3,
                reference = np.nanmin(layers[j]['grid'].values), # bottom of prisms is 
                # the deepest depth
                properties={'density':layers[j]['grid'].density})
            print(f'{j} top: {int(np.nanmean(layers[j]["prisms"].top.values))}m and ' 
                f'bottom: {int(np.nanmean(layers[j]["prisms"].bottom.values))}m')
        else:
            # if spacing of layer doesn't match below layer's spacing, sample lower 
            # layer to get values for bottoms of prisms.
            if layers[j]['spacing'] != layers[reversed_layers_list.iloc[i-1]]['spacing']:
                print(f"resolutions don't match for {j} ({layers[j]['spacing']}m) and "
                    f"{reversed_layers_list.iloc[i-1]} "
                    f"({layers[reversed_layers_list.iloc[i-1]]['spacing']}m)")
                print(f"sampling {reversed_layers_list.iloc[i-1]} at"
                    f" {j} prism locations")
                tmp = layers[j]['grid'].to_dataframe().reset_index()
                tmp_regrid = pygmt.grdtrack(points = tmp[['x','y']], 
                                grid = layers[reversed_layers_list.iloc[i-1]]['grid'], 
                                newcolname = 'z_regrid', verbose='q')
                tmp['z_low']=tmp.merge(tmp_regrid, how = 'left', 
                    on = ['x','y']).z_regrid
                tmp_grd = pygmt.xyz2grd(tmp[['x','y','z_low']], region = buffer_region, 
                    registration='p', spacing = layers[j]['spacing'])

                layers[j]['prisms']=hm.prism_layer(
                    coordinates=(layers[j]['grid'].x.values, 
                        layers[j]['grid'].y.values),   
                    surface=layers[j]['grid'], 
                    reference=tmp_grd,
                    properties={'density':layers[j]['grid'].density})
                print(f'{j} top: {int(np.nanmean(layers[j]["prisms"].top.values))}m and' 
                    f' bottom: {int(np.nanmean(layers[j]["prisms"].bottom.values))}m')
            else:
                layers[j]['prisms']=hm.prism_layer(
                    coordinates=(layers[j]['grid'].x.values, 
                        layers[j]['grid'].y.values),   
                    surface=layers[j]['grid'], 
                    reference=layers[reversed_layers_list.iloc[i-1]]['grid'],
                    properties={'density':layers[j]['grid'].density})
                print(f'{j} top: {int(np.nanmean(layers[j]["prisms"].top.values))}m and' 
                    f' bottom: {int(np.nanmean(layers[j]["prisms"].bottom.values))}m')
    
    if plot == True:
        if plot_type=='3D':
            # plot prisms layers in 3D with pyvista
            plotter = pv.Plotter(lighting="three_lights", window_size=(5000, 5000))
            colors = ['lavender','aqua','goldenrod','saddlebrown','black']
            for i, j in enumerate(kwargs.get('layers_for_3d', 
                pd.Series([k for k,v in layers.items()]))):
                prisms = layers[j]['prisms'].rio.set_spatial_dims(
                        'easting', 'northing').rio.write_crs("epsg:3031").rio.clip_box(
                                minx=plot_region[0], maxx=plot_region[1], 
                                miny=plot_region[2], maxy=plot_region[3])
                layers[j]['pvprisms'] = prisms.prism_layer.to_pyvista()
                if kwargs.get('clip_cube', True) == True:
                    # to clip out a cube
                    bounds = [
                        plot_region[0], 
                        plot_region[0]+((plot_region[1]-plot_region[0])/2),
                        plot_region[2], 
                        plot_region[2]+((plot_region[3]-plot_region[2])/2), 
                        -50e3, 1e3]
                    layers[j]['pvprisms'] = layers[j]['pvprisms'].clip_box(bounds, 
                        invert=True)
                plotter.add_mesh(
                    layers[j]['pvprisms'], color=colors[i], 
                    # scalars="density", cmap='viridis', flip_scalars=True,
                    # smooth_shading=True, style='points', point_size=2, 
                    # show_edges=False, # for just plotting surfaces
                    smooth_shading=True, style='surface', show_edges=False,
                )  
            plotter.set_scale(zscale=20)  # exaggerate the vertical coordinate
            plotter.camera_position = "xz"
            plotter.camera.elevation = 20
            plotter.camera.azimuth = -25
            plotter.camera.zoom(1)
            plotter.show_axes()
            plotter.show()
        elif plot_type=='2D':
            for i, j in enumerate(layers):
                if i == 0:
                    fig, ax = plt.subplots(ncols=len(layers), nrows=1, figsize=(20,20))
                thick = (layers[j]['prisms'].top - layers[j]['prisms'].bottom)
                thick = thick.rio.set_spatial_dims(
                        'easting', 'northing').rio.write_crs("epsg:3031").rio.clip_box(
                                    minx=plot_region[0], maxx=plot_region[1], 
                                    miny=plot_region[2], maxy=plot_region[3])
                thick.plot(ax=ax[i], robust=True, cmap='viridis',
                    cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})
                ax[i].set_title(f'{j} prism thickness')
            for a in ax:
                if inversion_region is not None:
                    a.add_patch(mpl.patches.Rectangle(
                        xy = (inversion_region[0], inversion_region[2]), 
                        width = (inversion_region[1]-inversion_region[0]),
                        height = (inversion_region[3]-inversion_region[2]),
                        linewidth=1, fill=False))
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_xlabel('')
                a.set_ylabel('')
                a.set_aspect('equal')

def forward_grav_layers(
    layers: dict, 
    gravity: pd.DataFrame or xr.Dataset,  
    plot: bool = True, 
    plot_dists: bool = False,
    **kwargs
    ):
    """
    Calculate forward gravity of layers of prisms.

    Parameters
    ----------
    layers : dict
        Nested dict; where each layer is a dict with keys: 
            'spacing': int, float; grid spacing 
            'fname': str; grid file name
            'rho': int, float; constant density value for layer
            'grid': xarray.DataArray;  
            'df': pandas.DataFrame; 2d representation of grid
    gravity : pd.DataFrame or xr.DataSet, 
        locations to calculate forward gravity at. needs variables 'x', 'y', and 'z'.
    plot : bool, optional
        Choose whether to plot, by default True.
    plot_dists : bool, optional
        Choose whether to plot the resulting distributions, by default False.
    Keyword Args
    ------------
    grav_spacing : int
        spacing of the gravity data for create plots. 
    exclude_layers: np.ndarray
        list of layers to exclude from total forward gravity, useful for 
        applying excluded layer[s] forward gravity in Bouguer correction
    inversion_region: np.ndarray or str
        GMT format region for the inverion, by default is extent of gravity data
    plot_type: str
        Choose to use 'xarray' or 'pygmt' to plot the results, by default is 
        'xarray'
    power_spectrum: bool
            choose to plot radially average power spectrum of layers, by defaul is False

    Returns
    -------
    pd.DataFrame
        Returns the input dataframe with forward gravity of individual and combined 
        layers 
    """    
    grav_spacing = kwargs.get("grav_spacing", None)
    exclude_layers = kwargs.get("exclude_layers", [])
    plot_type = kwargs.get("plot_type", "xarray")
    
    if plot is True and grav_spacing is None:
        raise ValueError(
            f"If plot = {plot}, grav_spacing must be set."
            )

    # if inversion region not given, get from input grav data
    if isinstance(gravity, pd.DataFrame):
        print('using supplied DataFrame for observation points')
        inversion_region = kwargs.get('inversion_region', 
            vd.get_region((gravity.x, gravity.y)))
        df_forward = gravity.copy()
    elif isinstance(gravity, xr.Dataset):
        print("converting 'gravity' from xr.Dataset to pd.DataFrame")
        inversion_region = kwargs.get('inversion_region', [int(pygmt.grdinfo('tmp_outputs/tmp_grav.nc', 
            per_column="n", o=i)[:-1]) for i in range(4)])
        df_forward = vd.grid_to_table(gravity)#.dropna(subset='Gobs')
    else:
        raise TypeError("parameter 'gravity' should be pd.DataFrame or xr.Dataset")

    # remove excluded layers
    include_forward_layers = pd.Series([k for k, v in layers.items() if k not in 
        exclude_layers])
    
    # calculate forward gravity
    for k, v in layers.items():
        df_forward[f"{k}_forward_grav"] = v['prisms'].prism_layer.gravity(
            coordinates = (df_forward.x, df_forward.y, df_forward.z), 
            field = 'g_z')
        df_forward[f"{k}_forward_grav"] -= df_forward[f"{k}_forward_grav"].mean()
        print(f"{len(v['prisms'].to_dataframe().reset_index())} prisms in {k} layer")
        print(f'finished {k} layer')

    # add gravity effects of all input layers
    grav_layers_list = [f'{i}_forward_grav' for i in include_forward_layers]
    df_forward['forward_total'] = df_forward[grav_layers_list].sum(axis=1, skipna=True)

    if plot==True:
        if plot_type=='pygmt':
            utils.plot_grd(
                # plot_region=buffer_reg,
                grid=grid_grav.forward_total,
                cmap='jet',
                grd2cpt_name='grav',
                cbar_label = f"combined forward gravity (mGal)",)

            for i, j in enumerate(include_forward_layers):
                utils.plot_grd(
                    # plot_region=buffer_reg,
                    grid=grid_grav[j],
                    cmap = 'jet',
                    grd2cpt_name = 'grav',
                    cbar_label = f"{j} forward gravity (mGal)",
                    origin_shift='xshift',)
            fig.show(width=1200) 
        elif plot_type=='xarray':
            fig, ax = plt.subplots(ncols=len(include_forward_layers)+1, nrows=1, figsize=(20,20))
            grid= pygmt.xyz2grd(data=df_forward[['x','y','forward_total']], 
                    region=inversion_region, 
                    spacing=grav_spacing,  
                    registration='p')
            grid.plot(ax=ax[0], x='x', y='y', robust=True, cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax[0].set_title('Total forward gravity')
            for i, j in enumerate(include_forward_layers):
                grid = pygmt.xyz2grd(data=df_forward[['x','y',f"{j}_forward_grav"]], 
                        region=inversion_region, 
                        spacing=grav_spacing,  
                        registration='p')
                grid.plot(ax=ax[i+1], x='x', y='y', robust=True, cmap='RdBu_r',
                    cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})
                ax[i+1].set_title(f'{j} forward gravity')
            for a in ax:
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_xlabel('')
                a.set_ylabel('')
                a.set_aspect('equal')
    
    if plot_dists==True:
        # get column to include in plots
        df=df_forward[['forward_total']+grav_layers_list]
        # df = df_forward.drop(['x','y','z','Gobs'], axis=1)
        # reorganize
        data = df.melt(var_name='layer')
        # layout the subplot grid
        g = sns.FacetGrid(data, col="layer", sharex=False, sharey=False)#, col_wrap=5)

        # add histogram for each column
        # g.map(sns.histplot, 'value', binwidth=1, kde=True,)
        g.map(sns.kdeplot, 'value', fill=True, bw_adjust=.4)

        # groupby to get mean and median
        data_g = data.groupby('layer').agg(['mean', 'median'])

        # extract and flatten the axes from the figure
        axes = g.axes.flatten()

        # iterate through each axes
        for ax in axes:
            # extract the species name
            grav_type = ax.get_title().split(' = ')[1]
            
            # select the data for the species
            d = data_g.loc[grav_type, :]

            # plot the lines
            ax.axvline(x=d.value['mean'], c='k', ls='-', lw=1.5, label='mean')
            ax.axvline(x=d.value['median'], c='grey', ls='--', lw=1.5, label='median')
            ax.legend(fontsize=8)

        # seperate plot of layers include in forward total
        df = df_forward[['forward_total']+grav_layers_list]
        plt.figure()
        sns.histplot(data=df, 
            palette='viridis',
            kde=True,
            # stat='count',
            multiple='stack', 
            element='step'
        )

    if kwargs.get('power_spectrum', False) is True:
        utils.raps(
            df_forward, 
            ['forward_total']+grav_layers_list,
            region=inversion_region, 
            spacing=grav_spacing,)

    return df_forward

def anomalies(
    layers: dict, 
    input_grav: pd.DataFrame,  
    grav_spacing: int,
    regional_method: str,
    plot: bool = True, 
    plot_dists: bool = False,
    **kwargs
    ): 
    """
    Calculate the residual gravity anomaly from 1 of 3 methods. Starting with the 
    misfit between observed and forward gravity, remove the regional misfit field to get
    the residual. The regional misfit if calculated from either:
    1) a user-defined degree 2D polynomail trend of the misfit,
    2) filtering the misfit with a user-defined filter, or
    3) interpolate the misfit at only constraint points.

    Parameters
    ----------
    layers : dict
        Nested dict; where each layer is a dict with keys: 
            'spacing': int, float; grid spacing 
            'fname': str; grid file name
            'rho': int, float; constant density value for layer
            'grid': xarray.DataArray;  
            'df': pandas.DataFrame; 2d representation of grid
    input_grav : pd.DataFrame
        _input gravity data
    grav_spacing : int
        spacing of gravity data, to use for make misfit grid and plotting
    regional_method : {'trend', 'filter', 'constraints'}
        choose a method to determine the regional gravity misfit.
    plot : bool, optional
        Choose whether to plot the resulting anomalies, by default True.
    plot_dists : bool, optional
        Choose whether to plot the distributions of the anomalies, by default False.

    Other Parameters
    ----------------
    **kwargs : dict
        input_grav_column: str,
            name of the column which contains the observed gravity, by default is 'Gobs'
        input_forward_column: str,
            name of the column which contains the total forward gravity, by default is 
            'forward_total'
        corrections: np.ndarray,
            list of layers to include in partial Bouguer correction of Observed 
            gravity data. 
        inversion_region: np.ndarray or str,
            GMT format region for the inverion, by default is extent of gravity data
        trend: int,
            trend order used from calculating regional misfit if 
            regional_method = 'trend'.
        filter: str, 
            input string for pygmt.grdfilter() for calculating regional misfit if 
            regional_method = 'filter', ex. "g200e3" gives a 200km Gaussian filter.
        constraints: pd.DataFrame, 
            Locations of constraint points to interpolate between for calculating 
            regional misfit if regional_method = 'constraints'.
        plot_constraints: bool,
            Choose to plot constraint locations, by default is True.
        constraints2plot: pd.DataFrame,
            if plotting constraints, supply locations, by default is all constraints.
        plot_type: {'xarray', 'pygmt'},
            Choose to use 'xarray' or 'pygmt' to plot the results, by default is 
            'xarray'
        fill_method" {'pygmt', 'rioxarray'},
            Choose method to fill nans, by default is 'pygmt'
        crs: str,
            if fill_method = 'rioxarray', set the coordinate reference system to be used
            in rioxarray.
        power_spectrum: bool
            choose to plot radially average power spectrum of layers, by defaul is False
    Returns
    -------
    pd.DataFrame
        Returns the input gravity dataframe with 4 additional columns:
        'grav_corrected': Observed gravity, or if 'corrections' is not None, then 
            Observered gravity with the forward gravity of the specified layers removed.
        'misfit': 'gravity_corrected' - total forward gravity
        'reg': regional gravity misfit 
        'res': residual graivty misift
    """    
    input_forward_column = kwargs.get("input_forward_column", "forward_total")
    input_grav_column = kwargs.get("input_grav_column", "Gobs")
    corrections = kwargs.get("corrections", None)
    # either set input inversion region or get from input gravity data extent
    inversion_region = kwargs.get('inversion_region', 
        vd.get_region((input_grav.x, input_grav.y)))
    
    trend = kwargs.get('trend', None)
    filter = kwargs.get('filter', None)
    constraints = kwargs.get('constraints', None)
    
    if kwargs.get(regional_method) is None:
        raise ValueError(
            f"Must provide keyword argument '{regional_method}' if regional_method ="
            f" {regional_method}."
            )

    plot_constraints = kwargs.get('plot_constraints', True)
    
    constraints2plot = kwargs.get('constraints2plot', constraints)
    plot_type = kwargs.get("plot_type", "xarray")
    
    if  (plot is True) and \
        (plot_constraints is True) and \
        (kwargs.get('constraints2plot') is None) and \
        (constraints is None):
        raise ValueError(
            f"If plot and plot_constraints are both True, must provide parameter 'constraints' or "
            f"'constriants2plot'."
            )
    crs = kwargs.get('crs', None)
    fill_method = kwargs.get('fill_method', 'pygmt')

    if fill_method == 'rioxarray' and crs is None:
        raise ValueError(
            f"Must provide 'crs' if fill_method = {fill_method}."
            )

    anomalies = input_grav.copy() 
    
    # if anomalies already calculated, drop the columns
    try:
        anomalies.drop(columns=['misfit','reg','res'], inplace=True)
    except:
        pass
    
    # apply partial Bouguer correction of layers in 'corrections'  
    if corrections is not None:
        for i, j in enumerate(corrections):
            anomalies['grav_corrected'] = anomalies[input_grav_column] - \
            anomalies[f"{j}_forward_grav"]
    else:
        print('no bouguer corrections applied')
        anomalies['grav_corrected'] = anomalies[input_grav_column]

    # get obs-forward misfit
    anomalies['misfit'] =  anomalies.grav_corrected - anomalies[input_forward_column]
    
    # fill misfit nans with 1 of 2 methods
    if fill_method == 'pygmt':
        """option 1) with pygmt.grdfill(), needs grav_spacing and inversion_region"""
        misfit = pygmt.xyz2grd(data=anomalies[['x','y','misfit']], 
                region=inversion_region, 
                spacing=grav_spacing,  
                registration='p')
        misfit_filled = pygmt.grdfill(misfit, mode='n')
    elif fill_method == 'rioxarray':
        """option 1) with rio.interpolate(), needs crs set."""
        misfit = anomalies.set_index(['y','x']).to_xarray().misfit.rio.write_crs(crs)
        misfit_filled = misfit.rio.write_nodata(np.nan).rio.interpolate_na()

    if regional_method=='trend':
        df = vd.grid_to_table(misfit_filled).astype('float64')
        trend = vd.Trend(degree=trend).fit((df.x, df.y.values), df.z)
        anomalies['reg'] = trend.predict((anomalies.x, anomalies.y))
        anomalies['res'] = anomalies.misfit - anomalies.reg

    elif regional_method=='filter':
        # filter the observed-forward misfit with the provided filter in meters
        regional_misfit = pygmt.grdfilter(
                    misfit, filter=filter, distance='0')
        # sample the results and merge into the anomalies dataframe
        tmp_regrid = pygmt.grdtrack(points = anomalies[['x','y']], 
            grid = regional_misfit, newcolname = 'reg', verbose='q')
        anomalies = anomalies.merge(tmp_regrid, on=['x','y'], how='left')
        anomalies['res'] = anomalies.misfit - anomalies.reg

    elif regional_method=='constraints': 
        # sample observed-forward misfit at constraint points
        tmp_regrid = pygmt.grdtrack(points = constraints[['x','y']], 
                            grid = misfit,
                            newcolname = 'misfit_sampled', verbose='q')
        # add sampled misfit value to each constraint location
        constraints['misfit']=constraints.merge(tmp_regrid, how = 'left', 
                on = ['x','y']).misfit_sampled
        # get median misfit of constraint poitns in each 1km cell 
        blocked = pygmt.blockmedian(data=constraints[['x','y','misfit']], 
                                    spacing="1000", region=inversion_region,
                                    registration='p')
        # grid the entire region misfit based just on the misfit at the constraints
        regional_misfit = pygmt.surface(data=blocked, region=inversion_region,
            spacing=grav_spacing, registration='p', verbose='q')
        # sample the resulting grid and add to anomalies dataframe
        tmp_regrid = pygmt.grdtrack(points = anomalies[['x','y']], 
            grid = regional_misfit, radius=0, newcolname = 'reg', verbose='q')
        anomalies = anomalies.merge(tmp_regrid, on=['x','y'], how='left')
        anomalies['res'] =  anomalies.misfit - anomalies.reg

    RMS = round(np.sqrt((anomalies.res**2).mean(skipna=True)),2)
    print(f'Root mean squared residual: {RMS}mGal')

    if plot==True:
        if plot_type=='pygmt':
            grid = anomalies.Gobs
            utils.plot_grd(grid=grid, cmap='jet', grd2cpt_name='grav',
                cbar_label = "observed gravity (mGal)",)

            grid = anomalies.Gobs_filt
            utils.plot_grd(grid=grid, cmap='plotting/grav.cpt', #grd2cpt_name='grav',
                cbar_label = "filtered observed gravity (mGal)", origin_shift='xshift',)

            grid = anomalies.forward_total
            utils.plot_grd(grid=grid, cmap='plotting/grav.cpt', #grd2cpt_name='grav',
                cbar_label = "forward gravity (mGal)", origin_shift='xshift',)

            grid = anomalies.misfit
            utils.plot_grd(grid=grid, cmap='polar+h0', grd2cpt_name='grav',
                cbar_label = "gravity misfit (mGal)", origin_shift='xshift',)

            grid = anomalies.misfit_filt
            utils.plot_grd(grid=grid, cmap='plotting/grav.cpt',
                cbar_label = "regional gravity misfit (mGal)", origin_shift='xshift',)
            fig.show(width=1200) 

        elif plot_type=='xarray':
            fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20,20))

            grid = pygmt.xyz2grd(data=anomalies[['x','y','grav_corrected']], 
                        region=inversion_region, 
                        spacing=grav_spacing,  
                        registration='p')
            grid.plot(ax=ax[0], x='x', y='y', robust=True, cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax[0].set_title('observed / partial bouguer gravity')
            
            grid = pygmt.xyz2grd(data=anomalies[['x','y','forward_total']], 
                    region=inversion_region, 
                    spacing=grav_spacing,  
                    registration='p')
            masked = utils.mask_from_shp("plotting/MEaSUREs_RIS.shp", xr_grid=grid, 
                masked=True, invert=False)
            lims = (-vd.maxabs(masked), vd.maxabs(masked))
            # lims = (np.nanquantile(masked, q=.05),
            #         np.nanquantile(masked, q=.95))
            grid.plot(ax=ax[1], x='x', y='y', vmin=lims[0], vmax=lims[1], cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax[1].set_title('forward gravity')

            grid = misfit
            grid.plot(ax=ax[2], x='x', y='y', robust=True, cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            if plot_constraints is True:
                ax[2].plot(constraints2plot.x, constraints2plot.y, 'k+')
            ax[2].set_title('gravity misfit')

            grid = pygmt.xyz2grd(data=anomalies[['x','y','reg']].dropna(), 
                        region=inversion_region, 
                        spacing=grav_spacing,  
                        registration='p')
            grid.plot(ax=ax[3], x='x', y='y', robust=True, cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})
            if plot_constraints is True:
                ax[3].plot(constraints2plot.x, constraints2plot.y, 'k+')  
            ax[3].set_title('regional misfit')

            grid = pygmt.xyz2grd(data=anomalies[['x','y','res']].dropna(), 
                        region=inversion_region, 
                        spacing=grav_spacing,  
                        registration='p')
            grid.plot(ax=ax[4], x='x', y='y', robust=True, cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            if plot_constraints is True:
                ax[4].plot(constraints2plot.x, constraints2plot.y, 'k+')
            ax[4].set_title('residual misfit')

            for a in ax:
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_xlabel('')
                a.set_ylabel('')
                a.set_aspect('equal')                
    
    if plot_dists==True:
        # get column to include in plots
        df=anomalies[['grav_corrected', 'forward_total', 'misfit', 'reg', 'res']]
        # reorganize
        data = df.melt(var_name='gravity type')
        # layout the subplot grid
        g = sns.FacetGrid(data, col="gravity type")#, sharex=False, sharey=False)#, col_wrap=5)

        # add histogram for each column
        # g.map(sns.histplot, 'value', binwidth=1, kde=True,)
        g.map(sns.kdeplot, 'value', fill=True, bw_adjust=.4)

        # groupby to get mean and median
        data_g = data.groupby('gravity type').agg(['mean', 'median'])

        # extract and flatten the axes from the figure
        axes = g.axes.flatten()

        # iterate through each axes
        for ax in axes:
            # extract the species name
            grav_type = ax.get_title().split(' = ')[1]
            
            # select the data for the species
            d = data_g.loc[grav_type, :]

            # plot the lines
            ax.axvline(x=d.value['mean'], c='k', ls='-', lw=1.5, label='mean')
            ax.axvline(x=d.value['median'], c='grey', ls='--', lw=1.5, label='median')
            ax.legend(fontsize=8)

        # seperate plot of misfit components
        df = anomalies[['grav_corrected', 'forward_total', 'misfit',]]
        plt.figure()
        sns.histplot(data=df, 
            palette='viridis',
            kde=True,
            # stat='count',
            multiple='stack', 
            element='step'
        )

    if kwargs.get('power_spectrum', False) is True:
        utils.raps(
            anomalies, 
            ['misfit', 'reg', 'res'],
            region=inversion_region, 
            spacing=grav_spacing,)   

    return anomalies

def density_inversion(density_layer, max_density_change=2000,  input_grav=None, 
    plot=True):
    """
    Function to invert gravity anomaly to update a prism layer's density.
    density_layer: str; layer to perform inversion on
    max_density_change: int, maximum amount to change each prisms density by, in kg/m^3
    input_grav: xarray.DataSet
    plot: bool; defaults to True
    """    
    if input_grav is None:
        input_grav=df_grav.Gobs_shift_filt
    # density in kg/m3
    forward_grav_layers(layers=layers, plot=False)
    spacing = layers[density_layer]['spacing']

    # df_grav['inv_misfit']=df_grav.Gobs_shift-df_grav[f'forward_grav_total']
    df_grav['inv_misfit']=input_grav - df_grav[f'forward_grav_total']

    # get prisms' coordinates from active layer
    prisms = layers[density_layer]['prisms'].to_dataframe().reset_index().dropna()

    print(f'active layer average density: {int(prisms.density.mean())}kg/m3')

    MAT_DENS = (np.zeros([len(input_grav),len(prisms)]))

    initial_RMS = round(np.sqrt((df_grav['inv_misfit'] **2).mean()),2)
    print(f"initial RMS = {initial_RMS}mGal")
    print('calculating sensitivity matrix to determine density correction')

    prisms_n=[]
    for x in range(len(layers[density_layer]['prisms'].easting.values)):
        for y in range(len(layers[density_layer]['prisms'].northing.values)):
            prisms_n.append(layers[density_layer]['prisms'].prism_layer.get_prism((x,y)))
    for col, prism in enumerate(prisms_n):
        MAT_DENS[:, col] = hm.prism_gravity(
            coordinates = (df_grav.x, df_grav.y, df_grav.z),
            prisms = prism,
            density = 1, # unit density
            field = 'g_z',)
    # Calculate shift to prism's densities to minimize misfit     
    Density_correction=lsqr(MAT_DENS,df_grav.inv_misfit,show=False)[0]

    # for i,j in enumerate((input_grav)): #add tqdm for progressbar
    #         MAT_DENS[i,:] = gravbox(
    #                             df_grav.y.iloc[i], df_grav.x.iloc[i], df_grav.z.iloc[i],
    #                             prisms.northing-spacing/2, prisms.northing+spacing/2,
    #                             prisms.easting-spacing/2, prisms.easting+spacing/2,
    #                             prisms.top,  prisms.bottom, np.ones_like(prisms.density))  # unit density, list of ones
    # # Calculate shift to prism's densities to minimize misfit     
    # Density_correction=lsqr(MAT_DENS,df_grav.inv_misfit,show=False)[0]*1000

    # apply max density change
    for i in range(0,len(prisms)):
        if Density_correction[i] > max_density_change:
            Density_correction[i]=max_density_change
        elif Density_correction[i] < -max_density_change:
            Density_correction[i]=-max_density_change

    # resetting the rho values with the above correction
    prisms['density_correction']=Density_correction
    prisms['updated_density']=prisms.density+prisms.density_correction
    dens_correction = pygmt.xyz2grd(x=prisms.easting, y=prisms.northing, z=prisms.density_correction, registration='p', 
                    region=buffer_reg, spacing=grav_spacing, projection=buffer_proj)
    dens_update = pygmt.xyz2grd(x=prisms.easting, y=prisms.northing, z=prisms.updated_density, registration='p', 
                    region=buffer_reg, spacing=layers[density_layer]['spacing'], projection=buffer_proj)
    initial_misfit = pygmt.xyz2grd(df_grav[['x','y','inv_misfit']], region=inv_reg, spacing=grav_spacing, registration='p')

    # apply the rho correction to the prism layer
    layers[density_layer]['prisms']['density'].values=dens_update.values
    print(f"average density: {int(layers[density_layer]['prisms'].to_dataframe().reset_index().dropna().density.mean())}kg/m3")
    # recalculate forward gravity of active layer               
    print('calculating updated forward gravity')
    df_grav[f'forward_grav_{density_layer}'] = layers[density_layer]['prisms'].prism_layer.gravity(
            coordinates=(df_grav.x, df_grav.y, df_grav.z),
            field = 'g_z')

    # Recalculate of gravity misfit, i.e., the difference between calculated and observed gravity
    df_grav['forward_grav_total'] = df_grav.forward_grav_total - \
                                    df_grav[f'{density_layer}_forward_grav'] + \
                                    df_grav[f'forward_grav_{density_layer}']
                                    
    df_grav.inv_misfit = input_grav - df_grav.forward_grav_total
    final_RMS = round(np.sqrt((df_grav.inv_misfit **2).mean()),2)
    print(f"RMSE after inversion = {final_RMS}mGal")
    final_misfit = pygmt.xyz2grd(df_grav[['x','y','inv_misfit']], region=buffer_reg, registration='p', spacing=grav_spacing)

    if plot==True:
        grid = initial_misfit
        utils.plot_grd(grid=grid, cmap='polar+h0', grd2cpt_name='misfit',
            cbar_label = f"initial misfit (mGal) [{initial_RMS}]")

        grid = dens_correction
        utils.plot_grd(grid=grid, cmap='polar+h0', grd2cpt_name='dens_corr',
            cbar_label = "density correction (kg/m3)", origin_shift='xshift')

        grid = dens_update
        utils.plot_grd(grid=grid, cmap='viridis', grd2cpt_name='dens_update',
            cbar_label = "updated density (kg/m3)", origin_shift='xshift')

        grid = final_misfit
        utils.plot_grd(grid=grid, cmap='polar+h0',# grd2cpt_name='misfit',
            cbar_label = f"final misfit (mGal) [{final_RMS}]", origin_shift='xshift')

        fig.show(width=1200)

def grav_column_der(x0,y0,z0,xc,yc,z1,z2,res,rho):
    """
    Function to calculate the vertical derivate of the gravitational acceleration cause by a right, rectangular prism.
    Approximate with Hammer's annulus approximation.
    x0, y0, z0: floats, coordinates of gravity observation points
    xc, yc, z1, z2: floats, coordinates of prism's y, x, top, and bottom, respectively. 
    res: float, resolution of prism layer in meters,
    rho: float, density of prisms, in kg/m^3
    """
    r=np.sqrt((x0-xc)**2+(y0-yc)**2)
    r1=r-0.5*res 
    r2=r+0.5*res   
    r1[r1<0]=0 # will fail if prism is under obs point
    r2[r1<0]=0.5*res
    f=res**2/(np.pi*(r2**2-r1**2)) #eq 2.19 in McCubbine 2016 Thesis
    anomaly_grad=0.0419*f*rho*(z1-z0)*(1/np.sqrt(r2**2+(z1-z0)**2)-1/np.sqrt(r1**2+(z1-z0)**2))
    return anomaly_grad

def jacobian_annular(gravity_data, gravity_col, prisms, spacing):
    """
    Function to calculate the Jacobian matrix using the annular cylinder approximation
    jacobian is matrix array with NG number of rows and NBath+NBase+NM number of columns
    uses vertical derivative of gravity to find least squares solution to minize gravity misfit for each grav station
    gravity_data: xarray.DataSet,
    gravity_col: str; variable of gravity_data for Observed gravity,
    prisms: pandas.DataFrame; dataframe of prisms coordinates with columns: 
        easting, northing, top, and bottom in meters.
    spacing: float, spacing of gravity data
    """
    # df = vd.grid_to_table(gravity_data).dropna(subset=gravity_col)
    df = gravity_data
    jac = np.empty((len(df[gravity_col]), len(prisms)), dtype=np.float64)
    for i,j in enumerate((df[gravity_col])):
            jac[i,:] = grav_column_der( # major issue here, way too slow
                df.y.iloc[i], # coords of gravity observation points
                df.x.iloc[i],
                df.z.iloc[i],  
                prisms.northing, prisms.easting,     
                prisms.top, 
                prisms.bottom,
                spacing,     
                prisms.density/1000)
    return jac

def jacobian_prism(gravity_data, gravity_col, model, delta, field):
    """
    Function to calculate the Jacobian matrix with the vertical gravity derivative 
    as a numerical approximation with small prisms 
    gravity_data: xarray.DataSet,
    gravity_col: str; variable of gravity_data for Observed gravity,
    model: xarray.DataSet; harmonica.prism_layer, with coordinates:
        easting, northing, top, and bottom,
        and variables: 'Density'.
    delta: float, size of small prism to add, in meters
    field: str; field to return, 'g_z' for gravitational acceleration. 
    """
    # df = vd.grid_to_table(gravity_data).dropna(subset=gravity_col)
    df = gravity_data
    jac = np.empty((len(df[gravity_col]), model.top.size), dtype=np.float64)
    # Build a generator for prisms (doesn't allocate memory,only returns at request)
    prisms_n_density = ( # about half of the cmp. time is here.
        (model.prism_layer.get_prism((i, j)), model.density.values[i, j])
        for i in range(model.northing.size)
        for j in range(model.easting.size)
    )
    for col, (prism,density) in enumerate(prisms_n_density):
        # Build a small prism ontop of existing prism (thickness equal to delta)
        bottom = prism[5] - delta / 2
        top = prism[5] + delta / 2
        delta_prism = (prism[0], prism[1], prism[2], prism[3], bottom, top)
        jac[:,col] = hm.prism_gravity( # other half of comp. time is here.
            coordinates=(df.x, df.y, df.z),
            prisms=delta_prism, 
            density=density, 
            field=field
            ) / delta 
    return jac
    # for x in range(len(layers[density_layer]['prisms'].easting.values)):
    #     for y in range(len(layers[density_layer]['prisms'].northing.values)):
    #         prisms_n.append(layers[density_layer]['prisms'].prism_layer.get_prism((x,y)))
    # for col, prism in enumerate(prisms_n):
    #     MAT_DENS[:, col] = hm.prism_gravity(
    #         coordinates = (df_grav.x, df_grav.y, df_grav.z),
    #         prisms = prism,
    #         density = 1, # unit density
    #         field = 'g_z',)

def solver(
    jacobian: np.array, 
    residuals: np.array, 
    solver_type='least squares',
    ):
    """
    Calculate shift to add to prism's for each iteration of the inversion.
    Parameters
    ----------
    jacobian : np.array
        input jacobian matrix with a row per gravity observation, and a column per prisms. 
    residuals : np.array
        array of gravity residuals
    solver_type : str, optional
        choose which solving method to use, by default 'least squares'

    Returns
    -------
    np.array
        array of corrrection values to apply to each prism.
    """
    if solver_type == 'least squares':
        # gives the amount that each column's Z1 needs to change by to have the smallest misfit
        # finds the least-squares solution to jacobian and Grav_Misfit, assigns the first value to Surface_correction
        step = lsqr(jacobian, residuals, show=False)[0] 
    # elif solver_type == 'gauss newton':
    #     # doesn't currently work
    #     hessian = jacobian.T @ jacobian
    #     gradient = jacobian.T @ residuals
    #     step = np.linalg.solve(hessian, gradient)
    # elif solver_type == 'steepest descent':
    #     # doesn't currently work
    #     step = - jacobian.T @ residuals
    return step

def plot_inversion_results(
    input_grav: Union[pd.DataFrame, str],
    iter_corrections: Union[pd.DataFrame, str],
    layers: dict,
    active_layer: str,
    grav_spacing: int,
    epsg: str,
    save_topo_nc: bool = False,
    save_residual_nc: bool = False,
    **kwargs
    ):
    """
    Plot the results of the inversion.

    Parameters
    ----------
    input_grav : pd.DataFrame or str
        Input gravity data with inversion results columns. Alternatively provide string 
        of csv filename.
    iter_corrections : pd.DataFrame or str
        Dataframe with corrections and updated geometry of the inversion layer for each 
        iteration. Alternatively provide string of csv filename.
    layers : dict
        Nested dict; where each layer is a dict with keys: 
            'spacing': int, float; grid spacing 
            'fname': str; grid file name
            'rho': int, float; constant density value for layer
            'grid': xarray.DataArray;  
            'df': pandas.DataFrame; 2d representation of grid
    active_layer : str
        Layer which was inverted for.
    grav_spacing : int
        Spacing of the gravity data for create plots.
    epsg : str
        Coordinate reference system string as input for rioxarray.clip().
    save_topo_nc : bool
        Choose to save the final inverted topography as a netcdf with an optional kwarg 
        filename.
    save_residual_nc : bool
        Choose to save the initial residual as a netcdf with an optional kwarg 
        filename.

    Keyword Args
    ----------------
    inversion_region : Union[str, np.ndarray], optional
          GMT format region for the inverion, by default is extent of gravity data
    plot_type: {'xarray', 'pygmt'},
            Choose to use 'xarray' or 'pygmt' to plot the results, by default is 
            'xarray'
    plot_region : Union[str, np.ndarray], optional
        GMT-format region to plot for both 2d and 3d plots, by default is inversion region
    plot_constraints: bool,
        Choose to plot constraint locations, by default is True.
    constraints: pd.DataFrame, 
        Locations of constraint points, by default is None.
    max_layer_change_per_iter : float
        Use the value set in inversion.geo_inversion(), by default is max absolute value of change in first iteration.
    topo_fname: str
        Customize the name of the saved netcdf file, by default is 'inverted_topo'.
    residual_fname: str
        Customize the name of the saved netcdf file, by default is 'initial_residual'.
    """    
 
    if isinstance(input_grav, str):
        input_grav = pd.read_csv(input_grav)

    if isinstance(iter_corrections, str):
        iter_corrections = pd.read_csv(iter_corrections)

    # either set input inversion region or get from input gravity data extent
    inversion_region = kwargs.get('inversion_region', 
        vd.get_region((input_grav.x, input_grav.y)))

    # if not supplied, set max correction equal to max absolute value of iter 1 correction
    max_abs = vd.maxabs(iter_corrections.iter_1_correction)
    max_layer_change_per_iter = kwargs.get('max_layer_change_per_iter', max_abs)

    plot_type = kwargs.get("plot_type", "xarray")
    plot_region = kwargs.get("plot_region", inversion_region)
    constraints = kwargs.get('constraints', None)
    plot_constraints = kwargs.get('plot_constraints', False)
    if  plot_constraints is True and constraints is None:
        raise ValueError(
            f"If plot_constraints is True, must provide parameter 'constraints'"
            )
    # pull columns from input dataframe
    initial_misfits = [s for s in input_grav.columns.to_list() if 'initial_misfit' in s]
    final_misfits = [s for s in input_grav.columns.to_list() if 'final_misfit' in s]
    forward_totals = [s for s in input_grav.columns.to_list() if '_forward_total' in s]
    iterations = [int(s[5:][:-15]) for s in initial_misfits]

    if plot_type=='pygmt':
        grid = initial_misfit
        if ITER ==1:
            utils.plot_grd(grid=grid, cmap='polar+h0', grd2cpt_name='misfit',
                cbar_label = f"initial misfit (mGal) [{initial_RMS}]", origin_shift='initialize')
        else:
            utils.plot_grd(grid=grid, cmap='plotting/misfit.cpt',
                cbar_label = f"initial misfit (mGal) [{initial_RMS}]", origin_shift='yshift')

        grid=layers[active_layer]['inv_grid']
        utils.plot_grd(grid=grid, cmap='globe',
            cbar_label = "updated bathymetry (m)", origin_shift='xshift')

        grid = iter_corr
        if ITER == 1:
            utils.plot_grd(grid=grid, cmap='polar+h0', grd2cpt_name='corr',
                cbar_label = "iteration correction (m)", origin_shift='xshift')
        else:
            utils.plot_grd(grid=grid, cmap='plotting/corr.cpt',
                cbar_label = "iteration correction (m)", origin_shift='xshift')

        grid = difference 
        if ITER ==1:
            utils.plot_grd(grid=grid, cmap='polar+h0', grd2cpt_name='diff',
                cbar_label = f"total {active_layer} difference (m)", origin_shift='xshift')
        else:
            utils.plot_grd(grid=grid, cmap='plotting/diff.cpt',
                cbar_label = f"total {active_layer} difference (m)", origin_shift='xshift')

        grid = input_grav.inv_misfit
        utils.plot_grd(grid=grid, cmap='plotting/misfit.cpt',
            cbar_label = f"final gravity misfit (mGal) [{final_RMS}]", origin_shift='xshift')
    
        # plot iteration label
        fig.shift_origin(xshift=((fig_width)/10))
        fig.text(projection = projection, 
            position='ML',
            justify='ML',
            text = f"It. #{ITER}",
            font = '30p,Helvetica,black',
            clearance = '+tO')
        fig.shift_origin(xshift=-((fig_width)/10))
        
        # shift back to origin 
        fig.shift_origin(xshift=-4*((fig_width + 2)/10))
        
    elif plot_type=='xarray':
        nrow = max(iterations)+3
        ncol = 3
        height = 5
        width = height * (input_grav.x.unique().size/input_grav.y.unique().size)

        initial_topo = layers[active_layer]['grid'].rio.set_spatial_dims(
                                'x','y').rio.write_crs(epsg).rio.clip_box(
                                    minx=plot_region[0], maxx=plot_region[1], 
                                    miny=plot_region[2], maxy=plot_region[3])

        grav = pygmt.xyz2grd(
                        data=input_grav[['x','y','grav_corrected']], 
                        region=plot_region, 
                        spacing=grav_spacing,  
                        registration='p', verbose='q')
        
        final_forward_total = pygmt.xyz2grd(
                        data=input_grav[['x','y',f'iter_{max(iterations)}_forward_total']], 
                        region=plot_region, 
                        spacing=grav_spacing,  
                        registration='p', verbose='q')

        initial_forward_total = pygmt.xyz2grd(
                        data=input_grav[['x','y','forward_total']], 
                        region=plot_region, 
                        spacing=grav_spacing,  
                        registration='p', verbose='q')

        maxabs = vd.maxabs(grav)*.95
        grav_lims = (-maxabs, maxabs)

        for ITER in iterations:
            initial_RMS = round(np.sqrt((input_grav[f"iter_{ITER}_initial_misfit"] \
                **2).mean(skipna=True)),2)
            final_RMS = round(np.sqrt((input_grav[f"iter_{ITER}_final_misfit"] \
                **2).mean(skipna=True).item()),2)
            correction = round(np.sqrt((iter_corrections[f"iter_{ITER}_correction"] \
                **2).mean(skipna=True).item()),2)
            
            grid1 = pygmt.xyz2grd(
                        data=input_grav[['x','y',f'iter_{ITER}_initial_misfit']], 
                        region=plot_region, 
                        spacing=grav_spacing,  
                        registration='p', verbose='q')

            grid2 = pygmt.xyz2grd(
                        data=iter_corrections[['x','y', f'iter_{ITER}_final_top']], 
                        region=plot_region, 
                        spacing=layers[active_layer]['spacing'],  
                        registration='p', verbose='q')

            grid3 = pygmt.xyz2grd(
                        data=iter_corrections[['x','y',f'iter_{ITER}_correction']], 
                        region=plot_region, 
                        spacing=grav_spacing,  
                        registration='p', verbose='q')
            
            if ITER==1:
                fig = plt.figure(figsize=(width*ncol, height*nrow)) 
                gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1]*ncol,
                                        wspace=0.1, hspace=0, 
                                        top=0.95, bottom=0.05, 
                                        left=0.17, right=0.845)
                misfit_lims = (-vd.maxabs(grid1)*.85, 
                                vd.maxabs(grid1)*.85)
                corr_lims = (-max_layer_change_per_iter, 
                                max_layer_change_per_iter)

                # diff_lims = (-vd.maxabs(grid4)*1.5, 
                #                 vd.maxabs(grid4)*1.5)
                # set active layer cmap to within ice shelf extent

                masked = utils.mask_from_shp("plotting/MEaSUREs_RIS.shp", xr_grid=grid2, 
                    masked=True, invert=False,)
                percent = 2
                topo_lims = (np.nanquantile(masked, q=percent/100),
                    np.nanquantile(masked, q=1-(percent/100)))

                initial_residual = grid1.copy()

            p=0

            ax = plt.subplot(gs[ITER-1,p])
            plt.text(-0.1, 0.5, f'Iteration #{ITER}',
                    transform=ax.transAxes,
                    rotation='vertical',
                    ha='center',
                    va='center',
                    fontsize=20,
                    )
            grid1.plot(ax=ax, x='x', y='y', vmin=misfit_lims[0], vmax=misfit_lims[1],
                cmap='RdBu_r', add_labels=False,
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(f'initial misfit: RMS={initial_RMS}mGal')
            if plot_constraints is True:
                ax.plot(constraints.x, constraints.y, 'k+')
            p+=1

            ax = plt.subplot(gs[ITER-1,p])
            grid2.plot(ax=ax, x='x', y='y', robust=True,#vmin=topo_lims[0], vmax=topo_lims[1],
                cmap='gist_earth', add_labels=False, 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax.set_xticklabels([])
            ax.set_yticklabels([])                
            ax.set_title('updated bathymetry')
            if plot_constraints is True:
                ax.plot(constraints.x, constraints.y, 'r+')
            p+=1

            ax = plt.subplot(gs[ITER-1,p])
            grid3.plot(ax=ax, x='x', y='y', vmin=corr_lims[0], vmax=corr_lims[1],
                cmap='RdBu_r', add_labels=False,
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(f'iteration correction: RMS={correction}m')
            if plot_constraints is True:
                ax.plot(constraints.x, constraints.y, 'k+')  
            final_topo = grid2.copy()

        ax = plt.subplot(gs[nrow-3, 0])
        initial_topo.plot(ax=ax, x='x', y='y', robust=True,#vmin=topo_lims[0], vmax=topo_lims[1],
            cmap='gist_earth', add_labels=False, 
            cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
        ax.set_xticklabels([])
        ax.set_yticklabels([])                
        ax.set_title(f'Initial {active_layer} topography')

        ax = plt.subplot(gs[nrow-3, 1])
        final_topo.plot(ax=ax, x='x', y='y', robust=True,#vmin=topo_lims[0], vmax=topo_lims[1],
            cmap='gist_earth', add_labels=False, 
            cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
        ax.set_xticklabels([])
        ax.set_yticklabels([])                
        ax.set_title(f'Final {active_layer} topography')

        ax = plt.subplot(gs[nrow-3, 2])
        (initial_topo-final_topo).plot(ax=ax, x='x', y='y', robust=True,
            cmap='RdBu_r', add_labels=False, 
            cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
        ax.set_xticklabels([])
        ax.set_yticklabels([])                
        ax.set_title('Difference')
        if plot_constraints is True:
            ax.plot(constraints.x, constraints.y, 'k+')

        ax = plt.subplot(gs[nrow-2, 0])
        grav.plot(ax=ax, x='x', y='y', 
            vmin=grav_lims[0], vmax=grav_lims[1],
            cmap='RdBu_r', add_labels=False, 
            cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
        ax.set_xticklabels([])
        ax.set_yticklabels([])                
        ax.set_title('Observed gravity')
        if plot_constraints is True:
            ax.plot(constraints.x, constraints.y, 'k+')

        ax = plt.subplot(gs[nrow-2, 1])
        initial_forward_total.plot(ax=ax, x='x', y='y', 
            vmin=grav_lims[0], vmax=grav_lims[1],
            cmap='RdBu_r', add_labels=False, 
            cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
        ax.set_xticklabels([])
        ax.set_yticklabels([])                
        ax.set_title('Initial forward gravity')
        if plot_constraints is True:
            ax.plot(constraints.x, constraints.y, 'k+')

        ax = plt.subplot(gs[nrow-2, 2])
        final_forward_total.plot(ax=ax, x='x', y='y', 
            vmin=grav_lims[0], vmax=grav_lims[1],
            cmap='RdBu_r', add_labels=False, 
            cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
        ax.set_xticklabels([])
        ax.set_yticklabels([])                
        ax.set_title('Final forward gravity')
        if plot_constraints is True:
            ax.plot(constraints.x, constraints.y, 'k+')
        
        ax = plt.subplot(gs[nrow-1, 0])
        (grav-initial_forward_total).plot(ax=ax, x='x', y='y', robust=True,
            cmap='RdBu_r', add_labels=False, 
            cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
        ax.set_xticklabels([])
        ax.set_yticklabels([])                
        ax.set_title('Initial misfit')
        if plot_constraints is True:
            ax.plot(constraints.x, constraints.y, 'k+')
        
        ax = plt.subplot(gs[nrow-1, 1])
        (grav-final_forward_total).plot(ax=ax, x='x', y='y', robust=True,
            cmap='RdBu_r', add_labels=False, 
            cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
        ax.set_xticklabels([])
        ax.set_yticklabels([])                
        ax.set_title('Final misfit')
        if plot_constraints is True:
            ax.plot(constraints.x, constraints.y, 'k+')

        if save_topo_nc is True:
            final_topo.to_netcdf(
                f"results/{kwargs.get('topo_fname','inverted_topo')}.nc")

        if save_residual_nc is True:
            initial_residual.to_netcdf(
                f"results/{kwargs.get('residual_fname','initial_residual')}.nc")

        
    if plot_type=='pygmt':
            fig.show(width=1200)

def geo_inversion(
    active_layer: str,
    layers: dict, 
    input_grav: pd.DataFrame,
    regional_method: str,
    misfit_sq_tolerance: float =0.00001,
    delta_misfit_squared_tolerance: float =0.002,
    Max_Iterations: int =3,
    deriv_type: str = 'prisms',
    solver_type: str = 'least squares',
    max_layer_change_per_iter: float=100,
    save_results: bool = False,
    **kwargs
    ): 
    """
    Invert geometry of upper surface of prism layer based on gravity anomalies.

    Parameters
    ----------
    active_layer : str
        layer to invert.
    layers : dict
        Nested dict; where each layer is a dict with keys: 
            'spacing': int, float; grid spacing 
            'fname': str; grid file name
            'rho': int, float; constant density value for layer
            'grid': xarray.DataArray;  
            'df': pandas.DataFrame; 2d representation of grid
    regional_method : {'trend', 'filter', 'constraints'}
        choose a method to determine the regional gravity misfit.
    input_grav : pd.DataFrame
        input gravity data with anomaly columns
    misfit_sq_tolerance : float, optional
        _description_, by default 0.00001
    delta_misfit_squared_tolerance : float, optional
        _description_, by default 0.002
    Max_Iterations : int, optional
        Terminate the inversion after this number of iterations, by default 3
    deriv_type : {'prisms', 'annulus'}, optional
        choose method for calculating vertical derivative of gravity, by 
        default 'prisms'
    solver_type : {'least squares', 'gauss newton', 'steepest descent'}, optional
        choose method for solving for geometry correction, by 
        default 'least squares'
    max_layer_change_per_iter : float, optional
        maximum amount each prism's surface can change by during each iteration, 
        by default 100
    save_results : bool, optional
        choose whether to save results as a csv, by default is False
    Other Parameters
    ----------------
    **kwargs : dict
        input_grav_column : str,
            "Gobs"
        apply_constraints : bool,
            False
        constraints_grid : xr.DataArray,
            grid with values from 0-1 by which to multiple each iterations correction 
            values by, defaults to None
        exclude_layers: np.ndarray,
            list of layers to include in partial Bouguer correction of Observed 
            gravity data. 
        buffer_region : Union[str, np.ndarray], optional
            region including buffer zone, by default reads region from first grid layer
        grav_spacing : int
                spacing of the gravity data for create plots.
        trend: int,
            trend order used from calculating regional misfit if 
            regional_method = 'trend'.
        filter: str, 
            input string for pygmt.grdfilter() for calculating regional misfit if 
            regional_method = 'filter', ex. "g200e3" gives a 200km Gaussian filter.
        constraints: pd.DataFrame, 
            Locations of constraint points to interpolate between for calculating 
            regional misfit if regional_method = 'constraints'.   
        fname_topo: str
            set csv filename, by default is 'topo_results' 
        fname_gravity: str
            set csv filename, by default is 'gravity_results'

    Returns
    -------
    tuple
        Returns tuple:
            iter_corrections: pd.DataFrame with corrections and updated geometry of the inversion layer for each iteration.
            gravity: pd.DataFrame with new columns of inversion results
    """    
    input_grav_column = kwargs.get("input_grav_column", "Gobs")
    apply_constraints = kwargs.get('apply_constraints', False)
    constraints_grid = kwargs.get('constraints_grid', None)
    if apply_constraints is not False and constraints_grid is None:
        raise ValueError(
            f"If apply_constraints = {apply_constraints}, constraints_grid must be applied."
            )
    trend = kwargs.get("trend", None)
    filter = kwargs.get("filter", None)
    constraints = kwargs.get("constraints", None)
    inversion_region = kwargs.get('inversion_region', 
        vd.get_region((input_grav.x, input_grav.y)))

    exclude_layers = kwargs.get('exclude_layers', [])
    buffer_region = kwargs.get('buffer_region', None)
    grav_spacing = kwargs.get('grav_spacing', None)
    if buffer_region is None:
        buffer_region = [int(pygmt.grdinfo(
            list(layers.values())[0]['grid'], 
            per_column="n", o=i)[:-1]) for i in range(4)]

    include_forward_layers = pd.Series([k for k, v in layers.items() if k not in 
        exclude_layers])

    spacing = layers[active_layer]['spacing'] 
    misfit_squared_updated=np.Inf  # positive infinity
    delta_misfit_squared=np.Inf  # positive infinity
    ind = include_forward_layers[include_forward_layers == active_layer].index[0]
    ITER=0
    # while delta_misfit_squared (inf) is greater than 1 + least squares tolerance (0.02)
    while delta_misfit_squared > 1+delta_misfit_squared_tolerance:
        ITER+=1 
        print(f"##################################\niteration {ITER}")
        if ITER==1:
            gravity = input_grav.copy()
        else:
            gravity['res'] = gravity[f"iter_{ITER-1}_final_misfit"]

        initial_RMS = round(np.sqrt((gravity.res**2).mean(skipna=True)),2)
        print(f"initial RMS residual = {initial_RMS}mGal")

        # get prisms' coordinates from active layer and layer above
        prisms = layers[active_layer]['prisms'].to_dataframe().reset_index().dropna()
        prisms_above = layers[include_forward_layers[ind-1]]['prisms'].to_dataframe().reset_index().dropna()

        # calculate jacobian
        if deriv_type == 'annulus': # major issue with grav_column_der, way too slow
            jac = jacobian_annular(gravity, input_grav_column, prisms, spacing)
        elif deriv_type == 'prisms':
            jac = jacobian_prism(gravity, input_grav_column, layers[active_layer]['prisms'], 1, "g_z")
        else:
            print('not valid derivative type')  

        # Calculate correction for each prism's surface
        # Surface_correction=lsqr(jac, gravity.res.values, show=False)[0] 
        Surface_correction = solver(jac, gravity.res.values, solver_type=solver_type)

        for i in range(0,len(prisms)):
            if Surface_correction[i] > max_layer_change_per_iter:
                Surface_correction[i]=max_layer_change_per_iter
            elif Surface_correction[i] < -max_layer_change_per_iter:
                Surface_correction[i]=-max_layer_change_per_iter
        prisms['correction']=Surface_correction
        prisms_above['correction']=Surface_correction
        print(f"RMS layer correction {round(np.sqrt((Surface_correction**2).mean()),2)}m")
        # apply above surface corrections 
        if apply_constraints is True:
            prisms['constraints'] = constraints_grid.to_dataframe().reset_index().z
            prisms_above['constraints'] = constraints_grid.to_dataframe().reset_index().z
            prisms['correction'] = prisms.constraints * prisms.correction
            prisms_above['correction'] = prisms_above.constraints * prisms_above.correction
        else:
            print('constraints not applied')

        if ITER==1:
            iter_corrections = prisms.rename(columns={'easting':'x', 'northing':'y'}).copy()
        iter_corrections[f'iter_{ITER}_initial_top'] = prisms.top.copy()

        prisms.top += prisms.correction
        prisms_above.bottom += prisms_above.correction
        iter_corrections[f'iter_{ITER}_final_top'] = prisms.top.copy()

        # apply the z correction to the active prism layer and the above layer with Harmonica 
        prisms_grid = pygmt.xyz2grd(prisms[['easting','northing','top']], 
                region=buffer_region, registration='p', spacing=spacing)
        prisms_above_grid = pygmt.xyz2grd(prisms_above[['easting','northing','bottom']], 
                region=buffer_region, registration='p', spacing=spacing)
        layers[active_layer]['prisms'].prism_layer.update_top_bottom(
                        surface = prisms_grid, 
                        reference = layers[active_layer]['prisms'].bottom)
        layers[include_forward_layers[ind-1]]['prisms'].prism_layer.update_top_bottom(
                        surface = layers[include_forward_layers[ind-1]]['prisms'].top, 
                        reference = prisms_above_grid)
        
        gravity[f'iter_{ITER}_initial_misfit'] = gravity.res

        iter_corrections[f"iter_{ITER}_correction"] = prisms.correction.copy()
 
        print('calculating updated forward gravity')
        gravity[f"iter_{ITER}_{active_layer}_forward_grav"] = \
            layers[active_layer]['prisms'].prism_layer.gravity(
                coordinates=(gravity.x, gravity.y, gravity.z), field = 'g_z')
        gravity[f"iter_{ITER}_{include_forward_layers[ind-1]}_forward_grav"] = \
            layers[include_forward_layers[ind-1]]['prisms'].prism_layer.gravity(
                coordinates=(gravity.x, gravity.y, gravity.z), field = 'g_z')
        # center on 0
        gravity[f"iter_{ITER}_{active_layer}_forward_grav"] -= \
            gravity[f"iter_{ITER}_{active_layer}_forward_grav"].mean()
        gravity[f"iter_{ITER}_{include_forward_layers[ind-1]}_forward_grav"] -= \
            gravity[f"iter_{ITER}_{include_forward_layers[ind-1]}_forward_grav"].mean()
        
        # add updated layers' column names to list
        updated_layers = [active_layer, include_forward_layers[ind-1]]
        updated_layers_list = [f"iter_{ITER}_{i}_forward_grav" for i in updated_layers]
        # add unchanged layers (excluding corrections layers) column names to list
        unchanged_layers = include_forward_layers[~include_forward_layers.str.contains(f"{include_forward_layers[ind-1]}|{active_layer}")]
        unchanged_layers_list = [f"{i}_forward_grav" for i in unchanged_layers]
        # combined list of column names
        updated_forward = updated_layers_list + unchanged_layers_list 
        # recalculate forward gravity with dataframe column names
        gravity[f'iter_{ITER}_forward_total']=gravity[updated_forward].sum(axis=1, skipna=True)
        # center on 0
        gravity[f'iter_{ITER}_forward_total'] -= gravity[f'iter_{ITER}_forward_total'].mean()

        print('updating the misfits')
        gravity[f"iter_{ITER}_final_misfit"] = anomalies(
            layers, 
            gravity, 
            grav_spacing,
            regional_method,
            input_forward_column = f'iter_{ITER}_forward_total',
            plot=False,
            corrections=exclude_layers, 
            trend=trend,
            filter=filter,
            constraints=constraints,
            inversion_region=inversion_region,
            ).res

        # for first iteration, divide infinity by mean square of gravity residuals, inversion will stop once this gets to delta_misfit_squared_tolerance (0.02)
        misfit_sq=(gravity[f"iter_{ITER}_final_misfit"]**2).mean(skipna=True).item()
        delta_misfit_squared=misfit_squared_updated/misfit_sq
        misfit_squared_updated=misfit_sq # updated 

        layers[active_layer]['inv_grid']=prisms.rename(
            columns={'easting':'x', 'northing':'y'}
            ).set_index(['y','x']).to_xarray().top
     
        active_layer_total_difference = layers[active_layer]['inv_grid'] - layers[active_layer]['grid']

        if ITER == Max_Iterations:
            print(f"Inversion terminated after {ITER} iterations with least-squares norm={int(misfit_sq)} because maximum number of iterations ({Max_Iterations}) reached")
            break
        # if misfit_sq < misfit_sq_tolerance:
        #     print(f"Inversion terminated after {ITER} iterations with least-squares norm={int(misfit_sq)} because least-squares norm < {misfit_sq_tolerance}")
        #     break
      
    # end of inversion iteration WHILE loop
    if delta_misfit_squared < 1+delta_misfit_squared_tolerance:
        print("terminated - no significant variation in least-squares norm ")
    
    if save_results is True:
        iter_corrections.to_csv(f"results/{kwargs.get('fname_topo', 'topo_results')}.csv", index=False)
        gravity.to_csv(f"results/{kwargs.get('fname_gravity', 'gravity_results')}.csv", index=False)
    
    return iter_corrections, gravity