# The standard Python science stack
import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# geoscience packages
import pygmt
from pyproj import Transformer
import pyvista as pv
import geopandas as gpd
import harmonica as hm
import verde as vd
import xarray as xr
from tqdm import tqdm
import rioxarray
import warnings
warnings.filterwarnings('ignore', message="pandas.Int64Index")
warnings.filterwarnings('ignore', message="pandas.Float64Index")

def get_grid_info(grid):
    """
    Function to return spacing and region from grid.grid
    returns tuple of spacing (int) and region (1x4 array)
    """
    spacing = pygmt.grdinfo(grid, per_column='n', o=7)[:-1]
    region = [int(pygmt.grdinfo(grid, per_column='n', o=i)[:-1]) for i in range(4)]
    return spacing, region

def dd2dms(dd):
    """
    Function to convert decimal degrees to minutes, seconds.
    Modified from https://stackoverflow.com/a/10286690/18686384
    """
    is_positive = dd >= 0
    dd = abs(dd)
    minutes,seconds = divmod(dd*3600,60)
    degrees,minutes = divmod(minutes,60)
    degrees = degrees if is_positive else -degrees
    return (f"{int(degrees)}:{int(minutes)}:{seconds}")

def latlon_to_epsg3031(df, reg=False, input=['lon', 'lat'], output=['x', 'y'],):
    """
    Function to convert coordinates from EPSG:4326 WGS84 in decimal degrees to
    EPSG:3031 Antarctic Polar Stereographic in meters. 
    default input dataframe columns are 'lon' and 'lat'
    default output dataframe columns are 'x' and 'y'
    default returns a dataframe with x, y, lat, and lon
    if reg=True, returns a region in format [e, w, n, s]
    """
    transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
    df[output[0]], df[output[1]] = transformer.transform(df[input[1]].tolist(), df[input[0]].tolist())
    if reg==True:
        df=[df[output[0]].min(), df[output[0]].max(), df[output[1]].max(), df[output[1]].min()]
    return df

def epsg3031_to_latlon(df, reg=False, input=['x', 'y'], output=['lon', 'lat']):
    """
    Function to convert coordinates from EPSG:3031 Antarctic Polar Stereographic in meters to 
    EPSG:4326 WGS84 in decimal degrees.
    default input dataframe columns are 'x' and 'y'
    default output dataframe columns are 'lon' and 'lat'
    default returns a dataframe with x, y, lat, and lon
    if reg=True, returns a region in format [e, w, n, s]
    """
    transformer = Transformer.from_crs("epsg:3031", "epsg:4326")
    df[output[1]], df[output[0]] = transformer.transform(df[input[1]].tolist(), df[input[0]].tolist())
    if reg==True:
        df=[df[output[0]].min(), df[output[0]].max(), df[output[1]].min(), df[output[1]].max()]
    return df

def reg_str_to_df(input, names=['x','y']):
    """
    Function to convert GMT region string [e, w, n, s] to pandas dataframe with 4 coordinates
    input: array of 4 strings.
    names: defauts to 'x', 'y', output df column names
    """
    bl = (input[0], input[2])
    br = (input[1], input[2])
    tl = (input[0], input[3])
    tr = (input[1], input[3])
    df = pd.DataFrame(data=[bl, br, tl, tr], columns=(names[0], names[1]))
    return df

def GMT_reg_xy_to_ll(input):
    """
    Function to convert GMT region string [e, w, n, s] in EPSG:3031 to deg:min:sec
    input: array of 4 strings.
    """
    df = reg_str_to_df(input)
    df_proj = epsg3031_to_latlon(df,reg=True)
    output = [dd2dms(x) for x in df_proj]
    return output

def mask_from_shp(
    shapefile, 
    invert=True, xr_grid=None, grid_file=None, region=None, spacing=None, masked=False, crs='epsg:3031'):
    """
    Function to create a mask or a masked grid from area inside or outside of a shapefile.
    shapefile: str; path to .shp filename.
    invert: bool; mask inside or outside of shapefile, defaults to True.
    xr_grid: xarray.DataArray(); to use to define region, or to mask.
    grid_gile: str; path to a .nc grid file to use to define region or to mask.
    region: str or 1x4 array; use to make mock grid if no grids are supplied. GMT region string or 1x4 array [e,w,n,s]
    spacing: str or float; GMT spacing string or float to use to make a mock grid if none are supplied.
    crs: str; if grid is provided, rasterio needs to assign a coordinate reference system via an epsg code
    """
    shp = gpd.read_file(shapefile).geometry
    if xr_grid is None and grid_file is None:
        coords = vd.grid_coordinates(region=region, spacing=spacing, pixel_register=True)
        ds = vd.make_xarray_grid(coords, np.ones_like(coords[0]), dims=('y', 'x'), data_names='z')
        xds=ds.z.rio.write_crs(crs)
    elif xr_grid is not None: 
        xds = xr_grid.rio.write_crs(crs)
    elif grid_file is not None:
        xds = xr.load_dataarray(grid_file).rio.write_crs(crs)

    masked_grd = xds.rio.clip(shp.geometry, xds.rio.crs, drop=False, invert=invert)
    mask_grd = np.isfinite(masked_grd)

    if masked == True:
        output = masked_grd
    elif masked == False:
        output = mask_grd
    return output

def plot_grd(
        grid, 
        cmap : str, 
        cbar_label : str, 
        plot_region=None, 
        cmap_region=None, 
        coast=False,
        constraints=False,
        grd2cpt_name=False, 
        origin_shift='initialize',
        ):
    """
    Function to automate PyGMT plotting
    """
    global fig, projection
    if plot_region is None:
        plot_region = inv_reg
    if cmap_region is None:
        cmap_region = inv_reg
    if plot_region == buffer_reg:
        projection = buffer_proj
    elif plot_region == inv_reg:
        projection = inv_proj
    # initialize figure or shift for new subplot
    if origin_shift=='initialize':
        fig = pygmt.Figure()   
    elif origin_shift=='xshift':
        fig.shift_origin(xshift=(fig_width + 2)/10)
    elif origin_shift=='yshift':
        fig.shift_origin(yshift=(fig_height + 12)/10)

    # set cmap
    if grd2cpt_name:
        pygmt.grd2cpt(
            cmap=cmap, 
            grid=grid, 
            region=cmap_region, 
            background=True, 
            continuous=True,
            output=f'plotting/{grd2cpt_name}.cpt')
        cmap = f'plotting/{grd2cpt_name}.cpt'

    fig.grdimage(
        grid=grid,
        cmap=cmap,
        projection=projection, 
        region=plot_region,
        nan_transparent=True,
        frame=['+gwhite'])

    fig.colorbar(
        cmap=cmap, 
        position='jBC+jTC+h', 
        frame=f'x+l"{cbar_label}"')

    if coast==True:
        fig.plot(
                projection = projection, 
                region = plot_region,
                data = gpd.read_file('plotting/GroundingLine_Antarctica_v02.shp'), 
                pen = '1.2p,black', 
                verbose='q',)

    fig.plot(data = gpd.read_file('plotting/Coastline_Antarctica_v02.shp'), 
            pen = '1.2p,black',
            verbose='q',
            )
    if constraints==True:
        fig.plot(
                x = constraints_RIS_df.x, 
                y = constraints_RIS_df.y, 
                style = 'c1.2p',
                color = 'black',
                projection = projection,
                region = plot_region,)

    if plot_region==buffer_reg:
        fig.plot(
            x = [inv_reg[0], inv_reg[0], inv_reg[1], inv_reg[1], inv_reg[0]], 
            y = [inv_reg[2], inv_reg[3], inv_reg[3], inv_reg[2], inv_reg[2]], 
            pen = '2p,black', 
            projection = projection,
            region = plot_region,)

def import_layers(
            layers_list, 
            spacing_list, 
            rho_list, 
            fname_list,
            grav_spacing, 
            active_layer,
            buffer_region,
            inversion_region,
            plot_region=None,
            # grav_file='gravity_data/ant4d_gravity_epsg_5k.nc',
            # grav_file='gravity_data/rosetta.nc',
            # grav_file='gravity_data/rosetta_eq_source_up_continued_Gobs.nc',
            # grav_elev_file='gravity_data/rosetta_eq_source_up_continued_elev.nc',
            grav_file='gravity_data/rosetta_eq_source_up_continued.csv',
            constraints=False,
            constraints_file='constraints_grid/constraints_grid.nc',
            constraints_points='bathymetry_data/bedmachine_RIGGS.csv',
            plot=True,
            plot_type='xarray'):
    """
    Function to import layers and their properties.
    layers_list: list of strings,
    spacing_list: list of floats or ints,
    rho_list: list of floats or ints,
    fname_list: list of strings, grids should be netcdf, tiff's might work. Should be pixel registered.
    grav_spacing: float or int,
    active_layer: str,
    plot_region: nd.array of shape (1,4); [e,w,n,s] region to plot for both 2d and 3d plots, defaults to inversion region (without buffer)
    plot: bool; defaults to True
    plot_type: str; defaults to 'xarray' for simple, fast plots, can choose 'pygmt' for slower nicer looking plots.
    """
    # global grid_grav, layers, df_grav, buffer_inv_str
    if constraints==True:
        global constraints_grid, constraints_df, constraints_RIS_df
    if plot_region==None:
        plot_region=inversion_region

    # read and resample gravity grid
    df_grav=pd.read_csv(grav_file, index_col=False)
    df_grav = df_grav[vd.inside((df_grav.x, df_grav.y), inversion_region)]
    df_grav['Gobs'] -= df_grav.Gobs.mean()
    # filt_grav = pygmt.grdfilter(
    #                     grid=grav_file,
    #                     filter=f'g{grav_spacing}', 
    #                     distance='0',
    #                     # nans='r', # retains NaNs
    #                     registration='p',
    #                     verbose='q')                 
    # grid_grav = pygmt.grdsample(
    #                     grid=filt_grav,
    #                     region=inv_reg, 
    #                     registration='p', 
    #                     spacing=grav_spacing,
    #                     verbose='q').to_dataset(name='Gobs')
    # center on 0
    # grid_grav['Gobs'] -= grid_grav.Gobs.mean().item()
  
    # df_grav = grid_grav.to_dataframe().reset_index()
    # df_grav.rename(columns={'z':'Gobs'}, inplace=True)
    # df_grav.dropna(how='any', inplace=True)  

    # make nested dictionary for layers and properties
    layers = {j:{'spacing':spacing_list[i], 
            'fname':fname_list[i], 
            'rho':rho_list[i]} for i, j in enumerate(layers_list)}

    # read and resample layer grids, convert to dataframes
    for k, v in layers.items():
        if int(v['spacing']) > int(pygmt.grdinfo(v['fname'], per_column=True, o=7)[:-1]):
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
                        grid=constraints_file,
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
                        grid=constraints_file,
                        region=buffer_region, 
                        registration='p', 
                        spacing=spacing_list[layers_list[layers_list == active_layer].index[0]],
                        verbose='q') 

        v['df'] = v['grid'].to_dataframe().reset_index()
        v['df']['rho']=v['rho']
        v['df'].dropna(how='any', inplace=True)
        v['len']=len(v['df'].x) 

    # add gravity elevation
    # grid_grav['z'] = pygmt.grdtrack(points = vd.grid_to_table(grid_grav)[['x','y']], 
    #             grid = layers[layers_list[0]]['grid'],
    #             newcolname='z').set_index(['x','y']).to_xarray().z   
    # try:                
        # grid_grav['z'] = grid_grav.Gobs*0 + 1e3 
        # filt_elev = pygmt.grdfilter(
        #                     grid=grav_elev_file,
        #                     filter=f'g{grav_spacing}', 
        #                     distance='0',
        #                     # nans='r', # retains NaNs
        #                     registration='p',
        #                     verbose='q')
        # grid_grav['z'] = pygmt.grdsample(
        #                 grid=filt_elev,
        #                 region=inv_reg, 
        #                 registration='p', 
        #                 spacing=grav_spacing,
        #                 verbose='q')
              
    # except:
    #     print('no gravity elevation data')

    if constraints==True:
        constraints_df =  pd.read_csv(constraints_points, index_col=False)
        # gmt grdmask plotting/MEaSUREs_RIS.shp -Gplotting/RIS_GL_mask_outer.nc -I1000 -rp -R{buffer_reg_str} -NNaN/NaN/1
        # constraints_RIS_df = pygmt.select(data=constraints_df, gridmask='plotting/RIS_GL_mask_outer.nc')
        mask = mask_from_shp("plotting/MEaSUREs_RIS.shp", masked=True, invert=False, region=buffer_region, spacing=1e3,)
        mask.to_netcdf('tmp_mask.nc')
        constraints_RIS_df = pygmt.select(data=constraints_df, gridmask='tmp_mask.nc',)
        # shp = gpd.read_file('plotting/MEaSUREs_RIS.shp').geometry
        # src = rasterio.open('inversion_layers/bedmachine/BedMachine_bed.nc')
        # src
        # out_image, out_transform, w = rasterio.mask.raster_geometry_mask(src, shp)
        # show(out_image)

        # da = xr.DataArray(out_image, dims=['y','x'],)
        # da.plot(x='x', y='y')
        # masked = da * grid
    # print lengths
    for k, v in layers.items():
        print(f"{k}: {v['len']} points, elevations: {int(np.nanmax(v['grid']))}m to {int(np.nanmin(v['grid']))}m") 
    print(f'gravity: {len(df_grav)} points')   
    try:
        print(f'gravity avg. elevation: {int(df_grav.z.max())}')   
    except:
        pass
    if constraints==True:
        print(f'bathymetry control points:{len(constraints_df)}') 

    if plot==True:
        if plot_type=='pygmt':
            # get max and min of all layer grids
            # max_list=[]
            # min_list=[]
            # for i, (k, v) in enumerate(layers.items()):
            #     max_list.append(np.nanmax(v['grid']))
            #     min_list.append(np.nanmin(v['grid']))
            # pygmt.makecpt(cmap='earth+h0', series=[np.min(min_list)*.5, np.max(max_list)*.2], output='plotting/layer.cpt')
            plot_grd(
                grid = grid_grav.Gobs, 
                plot_region=plot_region, 
                cmap = "jet",
                grd2cpt_name = 'grav',
                cbar_label = "observed gravity (mGal)", 
                constraints = constraints,    
                )

            for i, (k, v) in enumerate(layers.items()):
                plot_grd(
                    grid=layers[k]['grid'], 
                    plot_region=plot_region, 
                    # cmap = "plotting/layer.cpt",
                    cmap='viridis',
                    grd2cpt_name = 'elevations',
                    cbar_label = f"{k} elevation (m)",
                    origin_shift='xshift',
                    )
            fig.show(width=1200)
        elif plot_type=='xarray':
            if constraints == True:
                extra=2
            else:
                extra=1
            fig, ax = plt.subplots(ncols=len(layers)+extra, nrows=1, figsize=(20,20))#, constrained_layout=True)
            p=0
            grid = pygmt.xyz2grd(data=df_grav[['x','y','Gobs']], 
                region=inversion_region, 
                spacing=grav_spacing,  
                registration='p')
            # grid = df_grav.set_index(['y','x']).to_xarray().Gobs
            grid.plot(ax=ax[p], robust=True, cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax[p].set_title('Observed gravity')
            p+=1
            if constraints == True:
                constr = constraints_grid.rio.set_spatial_dims(
                                        'x', 'y').rio.write_crs("epsg:3031").rio.clip_box(
                                            minx=inversion_region[0], maxx=inversion_region[1], 
                                            miny=inversion_region[2], maxy=inversion_region[3])
                constr.plot(ax=ax[p], robust=True, cmap='copper', 
                    cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
                ax[p].set_title('Constraints grid')
            for i, j in enumerate(layers):
                # if plot_region==buffer_region:
                #     grid = layers[j]['grid']
                # elif plot_region==inversion_region:    
                grid = layers[j]['grid'].rio.set_spatial_dims(
                                'x', 'y').rio.write_crs("epsg:3031").rio.clip_box(
                                    minx=plot_region[0], maxx=plot_region[1], 
                                    miny=plot_region[2], maxy=plot_region[3])
                percent = 1
                lims = (np.nanquantile(grid, q=percent/100),
                    np.nanquantile(grid, q=1-(percent/100)))
                grid.plot(ax=ax[i+extra], vmin=lims[0], vmax=lims[1], cmap='gist_earth',
                    cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})
                ax[i+extra].set_title(f'{j} elevation')
                if constraints == True:
                    if j == active_layer:
                        ax[i+extra].plot(constraints_RIS_df.x, constraints_RIS_df.y, 'r+')
            for a in ax:
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_xlabel('')
                a.set_ylabel('')
                a.set_aspect('equal')
    return layers, df_grav, constraints_df, constraints_RIS_df

def grids_to_prism_layers(
    layers, 
    plot_region=None,
    buffer_region=None,
    inversion_region=None,
    layers_for_3d=None, 
    plot=False, 
    plot_type='3D',
    clip_cube=True
    ):
    """
    Function to turn nested dictionary of grids into series of vertical prisms between each layer.
    layers: nested dict; where each layer is a dict with keys: 
        'spacing': int, float; grid spacing 
        'fname': str; grid file name
        'rho': int, float; constant density value for layer
        'grid': xarray.DataArray;  
        'df': pandas.DataFrame; 2d representation of grid
    layers_for_3d: list, tuple; layers to include in 3D plot, defaults to all.
    plot_region: nd.array of shape (1,4); [e,w,n,s] region to plot for both 2d and 3d plots, defaults to inversion region (without buffer)
    plot: bool; defaults to False
    plot_type: str; defaults to '3D' which uses pyvista (issues if running from server), can choose 'thickness' for 2D plot.
    clip_cube: bool, defaults to True, clips cube out of 3D plot to help with visualization
    """

    if plot_region is None:
        plot_region = inversion_region
    if layers_for_3d is None:
        # layers_for_3d = layers_list[:]
        layers_for_3d =  pd.Series([k for k,v in layers.items()])

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
                reference = np.nanmin(layers[j]['grid'].values), # bottom of prisms is the deepest depth
                properties={'density':layers[j]['grid'].density})
            print(f'{j} top: {int(np.nanmean(layers[j]["prisms"].top.values))}m and bottom: {int(np.nanmean(layers[j]["prisms"].bottom.values))}m')
        else:
            # if spacing of layer doesn't match below layer's spacing, sample lower layer to get values for bottoms of prisms.
            if layers[j]['spacing'] != layers[reversed_layers_list.iloc[i-1]]['spacing']:
                print(f"resolutions don't match for {j} ({layers[j]['spacing']}m) and {reversed_layers_list.iloc[i-1]} ({layers[reversed_layers_list.iloc[i-1]]['spacing']}m)")
                print(f"sampling {reversed_layers_list.iloc[i-1]} at {j} prism locations")
                tmp = layers[j]['grid'].to_dataframe().reset_index()
                tmp_regrid = pygmt.grdtrack(points = tmp[['x','y']], 
                                            grid = layers[reversed_layers_list.iloc[i-1]]['grid'], 
                                            newcolname = 'z_regrid', verbose='q')
                tmp['z_low']=tmp.merge(tmp_regrid, how = 'left', on = ['x','y']).z_regrid
                tmp_grd = pygmt.xyz2grd(tmp[['x','y','z_low']], region = buffer_region, registration='p', spacing = layers[j]['spacing'])

                layers[j]['prisms']=hm.prism_layer(
                    coordinates=(layers[j]['grid'].x.values, layers[j]['grid'].y.values),   
                    surface=layers[j]['grid'], 
                    reference=tmp_grd,
                    properties={'density':layers[j]['grid'].density})
                print(f'{j} top: {int(np.nanmean(layers[j]["prisms"].top.values))}m and bottom: {int(np.nanmean(layers[j]["prisms"].bottom.values))}m')
            else:
                layers[j]['prisms']=hm.prism_layer(
                    coordinates=(layers[j]['grid'].x.values, layers[j]['grid'].y.values),   
                    surface=layers[j]['grid'], 
                    reference=layers[reversed_layers_list.iloc[i-1]]['grid'],
                    properties={'density':layers[j]['grid'].density})
                print(f'{j} top: {int(np.nanmean(layers[j]["prisms"].top.values))}m and bottom: {int(np.nanmean(layers[j]["prisms"].bottom.values))}m')
    
    if plot == True:
        if plot_type=='3D':
            # plot prisms layers in 3D with pyvista
            plotter = pv.Plotter(lighting="three_lights", window_size=(5000, 5000))
            colors = ['lavender','aqua','goldenrod','saddlebrown','black']
            for i, j in enumerate(layers_for_3d):
                # if plot_region==buffer_reg:
                #     prisms  = layers[j]['prisms']
                # else:
                prisms = layers[j]['prisms'].rio.set_spatial_dims(
                            'easting', 'northing').rio.write_crs("epsg:3031").rio.clip_box(
                                minx=plot_region[0], maxx=plot_region[1], 
                                miny=plot_region[2], maxy=plot_region[3])
                layers[j]['pvprisms'] = prisms.prism_layer.to_pyvista()
                if clip_cube == True:
                    # to clip out a cube
                    bounds = [
                        plot_region[0], plot_region[0]+((plot_region[1]-plot_region[0])/2),
                        plot_region[2], plot_region[2]+((plot_region[3]-plot_region[2])/2), 
                        -50e3, 1e3]
                    layers[j]['pvprisms'] = layers[j]['pvprisms'].clip_box(bounds, invert=True)
                plotter.add_mesh(
                    layers[j]['pvprisms'], color=colors[i], #scalars="density", cmap='viridis', flip_scalars=True,
                    # smooth_shading=True, style='points', point_size=2, show_edges=False, # for just plotting surfaces
                    smooth_shading=True, style='surface', show_edges=False,# for 3D blocks
                )  
            plotter.set_scale(zscale=20)  # exaggerate the vertical coordinate
            plotter.camera_position = "xz"
            plotter.camera.elevation = 20
            plotter.camera.azimuth = -25
            plotter.camera.zoom(1)
            plotter.show_axes()
            plotter.show()
        elif plot_type=='thickness':
            for i, j in enumerate(layers):
                if i == 0:
                    fig, ax = plt.subplots(ncols=len(layers), nrows=1, figsize=(20,20))
                thick = (layers[j]['prisms'].top - layers[j]['prisms'].bottom)
                # if plot_region==buffer_reg:
                #     thick.plot(ax=ax[i], robust=True, cmap='viridis',
                #         cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})
                #     ax[i].set_title(f'{j} prism thickness')
                # elif plot_region==inv_reg:
                thick = thick.rio.set_spatial_dims(
                                'easting', 'northing').rio.write_crs("epsg:3031").rio.clip_box(
                                    minx=plot_region[0], maxx=plot_region[1], 
                                    miny=plot_region[2], maxy=plot_region[3])
                thick.plot(ax=ax[i], robust=True, cmap='viridis',
                    cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})
                ax[i].set_title(f'{j} prism thickness')
            for a in ax:
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_xlabel('')
                a.set_ylabel('')
                a.set_aspect('equal')

def forward_grav_layers(layers, gravity, exclude_layers, inversion_region, grav_spacing, plot=True, plot_type='xarray'):
    """
    Function to calculate forward gravity of layers of prisms.
    layers: nested dict; where each layer is a dict with keys: 
        'spacing': int, float; grid spacing 
        'fname': str; grid file name
        'rho': int, float; constant density value for layer
        'grid': xarray.DataArray;  
        'df': pandas.DataFrame; 2d representation of grid
    exclude_layers: list of strings; layers to exclude from total forward gravity.
    plot: bool; defaults to True
    plot_type: str; defaults to 'xarray' for simple, fast plots, can choose 'pygmt' for slower nicer looking plots.
    """
    df_forward = gravity.copy()
    # include_forward_layers = layers_list[~layers_list.isin(exclude_layers)]
    include_forward_layers = pd.Series([k for k, v in layers.items() if k not in exclude_layers])
    
    # Calculate inital forward gravity model of input layers
    # df_grav = vd.grid_to_table(grid_grav)
    
    for k, v in layers.items():
        df_forward[f"{k}_forward_grav"] = v['prisms'].prism_layer.gravity(
            coordinates = (df_forward.x, df_forward.y, df_forward.z), 
            field = 'g_z')
        df_forward[f"{k}_forward_grav"] -= df_forward[f"{k}_forward_grav"].mean()
        # grid_grav[k] = df_grav.set_index(['x','y']).to_xarray()[f"{k}_forward_grav"]
        # grid_grav[k] -= grid_grav[k].mean().item()
        print(f'finished {k} layer')

    # add gravity effects of all input layers
    grav_layers_list = [f'{i}_forward_grav' for i in include_forward_layers]
    df_forward['forward_total'] = df_forward[grav_layers_list].sum(axis=1, skipna=True)
    # grid_grav['forward_total'] = sum(d for d in grid_grav[include_forward_layers.values].data_vars.values())

    if plot==True:
        if plot_type=='pygmt':
            plot_grd(
                # plot_region=buffer_reg,
                grid=grid_grav.forward_total,
                cmap='jet',
                grd2cpt_name='grav',
                cbar_label = f"combined forward gravity (mGal)",)

            for i, j in enumerate(include_forward_layers):
                plot_grd(
                    # plot_region=buffer_reg,
                    grid=grid_grav[j],
                    cmap = 'jet',
                    grd2cpt_name = 'grav',
                    cbar_label = f"{j} forward gravity (mGal)",
                    origin_shift='xshift',)
            fig.show(width=1200) 
        elif plot_type=='xarray':
            fig, ax = plt.subplots(ncols=len(include_forward_layers)+1, nrows=1, figsize=(20,20))
            # grid = df_grav.set_index(['x','y']).to_xarray().forward_total
            grid= pygmt.xyz2grd(data=df_forward[['x','y','forward_total']], 
                    region=inversion_region, 
                    spacing=grav_spacing,  
                    registration='p')
            grid.plot(ax=ax[0], x='x', y='y', robust=True, cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax[0].set_title('Total forward gravity')
            for i, j in enumerate(include_forward_layers):
                # grid = df_grav.set_index(['x','y']).to_xarray()[f"{j}_forward_grav"]
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
    return df_forward

def anomalies(
    layers, 
    input_grav,
    input_grav_column,
    regional_method,
    grav_spacing,
    input_forward_column=None,
    corrections = None,
    filter='g200e3', 
    trend_order=8,
    plot=True,
    plot_type='xarray',
    inversion_region=None,
    constraints_df=None,
    constraints_RIS_df=None,
    ): 
    """
    Function to calculate gravity anomalies.
    layers: nested dict; where each layer is a dict with keys: 
        'spacing': int, float; grid spacing 
        'fname': str; grid file name
        'rho': int, float; constant density value for layer
        'grid': xarray.DataArray;  
        'df': pandas.DataFrame; 2d representation of grid
    input_grav: xarray.DataSet
    input_grav_column: str: variable of input_grav to use.
    regional_method: str; calculate anomaly with either 'trend' or 'filter' methods.
    corrections: list of strings; layers to remove from input gravity (remove ice for partial bouguer correction)
    filter: str; input string for pygmt.grdfilter()
    trend_order: int,
    plot: bool; defaults to True
    plot_type: str; defaults to 'xarray' for simple, fast plots, can choose 'pygmt' for slower nicer looking plots.
    """
    if input_forward_column is None:
        input_forward_column="forward_total"
    anomalies = input_grav.copy() 
    try:
        anomalies.drop(columns=['misfit','reg','res'], inplace=True)
    except:
        pass

    # remove forward grav of specified layers
    if corrections is not None:
        for i, j in enumerate(corrections):
            anomalies['grav_corrected'] = anomalies[input_grav_column] - anomalies[f"{j}_forward_grav"]
    else:
        print('no bouguer corrections applied')
        anomalies['grav_corrected'] = anomalies[input_grav_column]

    anomalies['misfit'] =  anomalies.grav_corrected - anomalies[input_forward_column]
    
    # fill nans for vd.Trend
    """option 1) with rio.interpolate(), needs crs set."""
    # misfit = anomalies.set_index(['y','x']).to_xarray().misfit.rio.write_crs('epsg:3031')
    # misfit_filled = misfit.rio.write_nodata(np.nan).rio.interpolate_na()
    """option 2) with pygmt.grdfill()"""
    # misfit = anomalies.set_index(['y','x']).to_xarray().misfit
    misfit = pygmt.xyz2grd(data=anomalies[['x','y','misfit']], 
                region=inversion_region, 
                spacing=grav_spacing,  
                registration='p')
    misfit_filled = pygmt.grdfill(misfit, mode='n')

    if regional_method=='filter':
        regional_misfit = pygmt.grdfilter(
                    misfit, filter=filter, distance='0')
        tmp_regrid = pygmt.grdtrack(points = anomalies[['x','y']], grid = regional_misfit,
                newcolname = 'reg', verbose='q')
        # anomalies['reg']=anomalies.merge(tmp_regrid, how = 'left', on = ['x','y']).sampled 
        anomalies = anomalies.merge(tmp_regrid, on=['x','y'], how='left')
        anomalies['res'] = anomalies.misfit - anomalies.reg

    elif regional_method=='trend':
        df = vd.grid_to_table(misfit_filled).astype('float64')
        trend = vd.Trend(degree=trend_order).fit((df.x, df.y.values), df.z)
        anomalies['reg'] = trend.predict((anomalies.x, anomalies.y))
        anomalies['res'] = anomalies.misfit - anomalies.reg

    elif regional_method=='constraints': # sample Gobs_misfit at constraint points
        tmp_regrid = pygmt.grdtrack(points = constraints_df[['x','y']], 
                                    grid = anomalies.set_index(['y','x']).to_xarray().misfit, 
                                    newcolname = 'misfit_sampled', verbose='q')
        constraints_df['misfit']=constraints_df.merge(tmp_regrid, how = 'left', on = ['x','y']).misfit_sampled
        blocked = pygmt.blockmedian(data=constraints_df[['x','y','misfit']], 
                                    spacing="1000", region=inversion_region)
        regional_misfit = pygmt.surface(data=blocked, region=inversion_region, spacing=grav_spacing, 
                    registration='p', verbose='q')
        tmp_regrid = pygmt.grdtrack(points = anomalies[['x','y']], grid = regional_misfit, radius=0,
                newcolname = 'reg', verbose='q')
        # print(anomalies.describe())
        anomalies = anomalies.merge(tmp_regrid, on=['x','y'], how='left')
        # anomalies['reg']=anomalies.merge(tmp_regrid, how = 'left', on = ['x','y']).sampled 
        anomalies['res'] =  anomalies.misfit - anomalies.reg

    if plot==True:
        if plot_type=='pygmt':
            grid = anomalies.Gobs
            plot_grd(grid=grid, cmap='jet', grd2cpt_name='grav',
                cbar_label = "observed gravity (mGal)",)

            grid = anomalies.Gobs_filt
            plot_grd(grid=grid, cmap='plotting/grav.cpt', #grd2cpt_name='grav',
                cbar_label = "filtered observed gravity (mGal)", origin_shift='xshift',)

            grid = anomalies.forward_total
            plot_grd(grid=grid, cmap='plotting/grav.cpt', #grd2cpt_name='grav',
                cbar_label = "forward gravity (mGal)", origin_shift='xshift',)

            grid = anomalies.misfit
            plot_grd(grid=grid, cmap='polar+h0', grd2cpt_name='grav',
                cbar_label = "gravity misfit (mGal)", origin_shift='xshift',)

            grid = anomalies.misfit_filt
            plot_grd(grid=grid, cmap='plotting/grav.cpt',
                cbar_label = "regional gravity misfit (mGal)", origin_shift='xshift',)
            fig.show(width=1200) 

        elif plot_type=='xarray':
            fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20,20))

            # grid =  anomalies.set_index(['y','x']).to_xarray().grav_corrected
            grid = pygmt.xyz2grd(data=anomalies[['x','y','grav_corrected']], 
                        region=inversion_region, 
                        spacing=grav_spacing,  
                        registration='p')
            grid.plot(ax=ax[0], x='x', y='y', robust=True, cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax[0].set_title('observed / partial bouguer gravity')
            
            # grid = anomalies.set_index(['y','x']).to_xarray().forward_total
            grid = pygmt.xyz2grd(data=anomalies[['x','y','forward_total']], 
                    region=inversion_region, 
                    spacing=grav_spacing,  
                    registration='p')
            masked = mask_from_shp("plotting/MEaSUREs_RIS.shp", xr_grid=grid, masked=True, invert=False)
            lims = (np.nanquantile(masked, q=.05),
                    np.nanquantile(masked, q=.95))
            grid.plot(ax=ax[1], x='x', y='y', vmin=lims[0], vmax=lims[1], cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax[1].set_title('forward gravity')

            grid = misfit
            grid.plot(ax=ax[2], x='x', y='y', robust=True, cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            # if regional_method == 'constraints':
            #     ax[2].plot(constraints_df.x, constraints_df.y, 'k+')
            ax[2].set_title('gravity misfit')

            # grid = anomalies.set_index(['y','x']).to_xarray().reg
            grid = pygmt.xyz2grd(data=anomalies[['x','y','reg']].dropna(), 
                        region=inversion_region, 
                        spacing=grav_spacing,  
                        registration='p')
            grid.plot(ax=ax[3], x='x', y='y', robust=True, cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})
            # if regional_method == 'constraints':
            #     ax[3].plot(constraints_df.x, constraints_df.y, 'k+')  
            ax[3].set_title('regional misfit')

            # grid = anomalies.set_index(['y','x']).to_xarray().res
            grid = pygmt.xyz2grd(data=anomalies[['x','y','res']].dropna(), 
                        region=inversion_region, 
                        spacing=grav_spacing,  
                        registration='p')
            grid.plot(ax=ax[4], x='x', y='y', robust=True, cmap='RdBu_r', 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            # if regional_method == 'constraints':
            #     ax[4].plot(constraints_df.x, constraints_df.y, 'k+')
            ax[4].set_title('residual misfit')

            for a in ax:
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_xlabel('')
                a.set_ylabel('')
                a.set_aspect('equal')                
    return anomalies

def density_inversion(density_layer, max_density_change=2000,  input_grav=None, plot=True):
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
        plot_grd(grid=grid, cmap='polar+h0', grd2cpt_name='misfit',
            cbar_label = f"initial misfit (mGal) [{initial_RMS}]")

        grid = dens_correction
        plot_grd(grid=grid, cmap='polar+h0', grd2cpt_name='dens_corr',
            cbar_label = "density correction (kg/m3)", origin_shift='xshift')

        grid = dens_update
        plot_grd(grid=grid, cmap='viridis', grd2cpt_name='dens_update',
            cbar_label = "updated density (kg/m3)", origin_shift='xshift')

        grid = final_misfit
        plot_grd(grid=grid, cmap='polar+h0',# grd2cpt_name='misfit',
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
            jac[i,:] = grav_column_der(
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
    df = vd.grid_to_table(gravity_data).dropna(subset=gravity_col)
    jac = np.empty((len(df[gravity_col]), model.top.size), dtype=np.float64)
    # Build a generator for prisms (doesn't allocate memory,only returns at request)
    prisms_n_density = ( 
        (model.prism_layer.get_prism((i, j)), model.density.values[i, j])
        for i in range(model.northing.size)
        for j in range(model.easting.size)
    )
    for col, (prism,density) in enumerate(prisms_n_density):
        # Build a small prism ontop of existing prism (thickness equal to delta)
        bottom = prism[5] - delta / 2
        top = prism[5] + delta / 2
        delta_prism = (prism[0], prism[1], prism[2], prism[3], bottom, top)
        jac[:,col] = hm.prism_gravity(
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


def plot_inversion_results(
    input_grav,
    inversion_region,
    active_layer,
    grav_spacing,
    epsg,
    max_layer_change_per_iter,
    constraints_RIS_df,
    layers,
    iter_corrections,
    constraints = False,
    plot_type='xarray',
    ):
    """
    Function to plot the results of the inversion
    """
    # pull columns from input dataframe
    initial_misfits = [s for s in input_grav.columns.to_list() if 'initial_misfit' in s]
    final_misfits = [s for s in input_grav.columns.to_list() if 'final_misfit' in s]
    forward_totals = [s for s in input_grav.columns.to_list() if '_forward_total' in s]
    iterations = [int(s[5:][:-15]) for s in initial_misfits]

    if plot_type=='pygmt':
        grid = initial_misfit
        if ITER ==1:
            plot_grd(grid=grid, cmap='polar+h0', grd2cpt_name='misfit',
                cbar_label = f"initial misfit (mGal) [{initial_RMS}]", origin_shift='initialize')
        else:
            plot_grd(grid=grid, cmap='plotting/misfit.cpt',
                cbar_label = f"initial misfit (mGal) [{initial_RMS}]", origin_shift='yshift')

        grid=layers[active_layer]['inv_grid']
        plot_grd(grid=grid, cmap='globe',
            cbar_label = "updated bathymetry (m)", origin_shift='xshift')

        grid = iter_corr
        if ITER == 1:
            plot_grd(grid=grid, cmap='polar+h0', grd2cpt_name='corr',
                cbar_label = "iteration correction (m)", origin_shift='xshift')
        else:
            plot_grd(grid=grid, cmap='plotting/corr.cpt',
                cbar_label = "iteration correction (m)", origin_shift='xshift')

        grid = difference 
        if ITER ==1:
            plot_grd(grid=grid, cmap='polar+h0', grd2cpt_name='diff',
                cbar_label = f"total {active_layer} difference (m)", origin_shift='xshift')
        else:
            plot_grd(grid=grid, cmap='plotting/diff.cpt',
                cbar_label = f"total {active_layer} difference (m)", origin_shift='xshift')

        grid = input_grav.inv_misfit
        plot_grd(grid=grid, cmap='plotting/misfit.cpt',
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
        nrow = max(iterations)+1
        ncol = 5
        height = 5
        width = height * (input_grav.x.unique().size/input_grav.y.unique().size)

        for ITER in iterations:
            if ITER==1:
                pass
            else:
                input_grav['res'] = input_grav[f"iter_{ITER-1}_final_misfit"]
            initial_RMS = round(np.sqrt((input_grav.res**2).mean(skipna=True)),2)
            final_RMS = round(np.sqrt((input_grav[f"iter_{ITER}_final_misfit"] **2).mean(skipna=True).item()),2)
            grid1 = pygmt.xyz2grd(data=input_grav[['x','y',f'iter_{ITER}_initial_misfit']], 
                        region=inversion_region, 
                        spacing=grav_spacing,  
                        registration='p')
            grid2 = layers[active_layer]['inv_grid'].rio.set_spatial_dims(
                                'x','y').rio.write_crs(epsg).rio.clip_box(
                                    minx=inversion_region[0], maxx=inversion_region[1], 
                                    miny=inversion_region[2], maxy=inversion_region[3])
            grid3 = iter_corrections.set_index(['x','y']).to_xarray()[f"iter_{ITER}_correction"].rio.set_spatial_dims(
                                'x','y').rio.write_crs(epsg).rio.clip_box(
                                    minx=inversion_region[0], maxx=inversion_region[1], 
                                    miny=inversion_region[2], maxy=inversion_region[3])
            active_layer_total_difference = layers[active_layer]['inv_grid'] - layers[active_layer]['grid']
            grid4 = active_layer_total_difference.rio.set_spatial_dims(
                                'x', 'y').rio.write_crs(epsg).rio.clip_box(
                                    minx=inversion_region[0], maxx=inversion_region[1], 
                                    miny=inversion_region[2], maxy=inversion_region[3])
            grid5 = pygmt.xyz2grd(data=input_grav[['x','y',f'iter_{ITER}_final_misfit']], 
                        region=inversion_region, 
                        spacing=grav_spacing,  
                        registration='p')
            final_topo = grid2
            initial_topo = layers[active_layer]['grid'].rio.set_spatial_dims(
                                'x','y').rio.write_crs(epsg).rio.clip_box(
                                    minx=inversion_region[0], maxx=inversion_region[1], 
                                    miny=inversion_region[2], maxy=inversion_region[3])
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
                diff_lims = (-vd.maxabs(grid4)*1.5, 
                                vd.maxabs(grid4)*1.5)
                # set active layer cmap to within ice shelf extent
                mask = xr.load_dataarray('plotting/RIS_mask.nc')
                masked = mask * grid2
                percent = 2
                topo_lims = (np.nanquantile(masked, q=percent/100),
                    np.nanquantile(masked, q=1-(percent/100)))
                # topo_lims = (-vd.maxabs(grid2), 
                #                 vd.maxabs(grid2))

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
            ax.set_title(f'initial misfit: {initial_RMS}mGal')
            p+=1

            ax = plt.subplot(gs[ITER-1,p])
            grid2.plot(ax=ax, x='x', y='y', vmin=topo_lims[0], vmax=topo_lims[1],
                cmap='gist_earth', add_labels=False, 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax.set_xticklabels([])
            ax.set_yticklabels([])                
            ax.set_title('updated bathymetry')
            if constraints == True:
                ax.plot(constraints_RIS_df.x, constraints_RIS_df.y, 'r+')
            p+=1

            ax = plt.subplot(gs[ITER-1,p])
            grid3.plot(ax=ax, x='x', y='y', vmin=corr_lims[0], vmax=corr_lims[1],
                cmap='RdBu_r', add_labels=False,
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title('iteration correction')
            p+=1

            ax = plt.subplot(gs[ITER-1,p])
            grid4.plot(ax=ax, x='x', y='y', vmin=diff_lims[0], vmax=diff_lims[1],
                cmap='RdBu_r', add_labels=False,
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(f'total {active_layer} difference')
            if constraints == True:
                ax.plot(constraints_RIS_df.x, constraints_RIS_df.y, 'k+')
            p+=1

            ax = plt.subplot(gs[ITER-1,p])
            grid5.plot(ax=ax, x='x', y='y', vmin=misfit_lims[0], vmax=misfit_lims[1],
                cmap='RdBu_r', add_labels=False,
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(f'final gravity misfit: {final_RMS}mGal')

            ax = plt.subplot(gs[ITER, 1])
            initial_topo.plot(ax=ax, x='x', y='y', vmin=topo_lims[0], vmax=topo_lims[1],
                cmap='gist_earth', add_labels=False, 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax.set_xticklabels([])
            ax.set_yticklabels([])                
            ax.set_title(f'Initial {active_layer} topography')

            ax = plt.subplot(gs[ITER, 2])
            final_topo.plot(ax=ax, x='x', y='y', vmin=topo_lims[0], vmax=topo_lims[1],
                cmap='gist_earth', add_labels=False, 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax.set_xticklabels([])
            ax.set_yticklabels([])                
            ax.set_title(f'Final {active_layer} topography')

            ax = plt.subplot(gs[ITER, 3])
            (initial_topo-final_topo).plot(ax=ax, x='x', y='y', robust=True,
                cmap='RdBu_r', add_labels=False, 
                cbar_kwargs={'orientation':'horizontal', 'anchor':(1,1.8)})  
            ax.set_xticklabels([])
            ax.set_yticklabels([])                
            ax.set_title('Difference')

    if plot_type=='pygmt':
            fig.show(width=1200)

def geo_inversion(
    active_layer,
    exclude_layers,
    layers, 
    input_grav,
    input_grav_column,
    regional_method,
    grav_spacing,
    inversion_region,
    buffer_region,
    filter='g200e3', 
    trend_order=2,
    # reset=True,
    constraints=False, 
    misfit_sq_tolerance=0.00001,
    delta_misfit_squared_tolerance=0.002,
    Max_Iterations=5,
    deriv_type = 'annulus',
    max_layer_change_per_iter=100, #meters
    ): 
    """
    Function to invert layer geometry based on gravity anomalies
    active_layer: str, layer to invert.
    exclude_layers: array of strings, layers to exclude from total forward gravity, and remove from observed gravity.
    layers: nested dict; where each layer is a dict with keys: 
        'spacing': int, float; grid spacing 
        'fname': str; grid file name
        'rho': int, float; constant density value for layer
        'grid': xarray.DataArray;  
        'df': pandas.DataFrame; 2d representation of grid
    input_grav: xarray.DataSet
    input_grav_column: str: variable of input_grav to use.
    regional_method: str; calculate anomaly with either 'trend' or 'filter' methods.
    filter: str; input string for pygmt.grdfilter()
    trend_order: int,
    plot: bool; defaults to True
    plot_type: str; defaults to 'xarray' for simple, fast plots, can choose 'pygmt' for slower nicer looking plots.
    reset: bool, defaults to True, recalculate original forward gravity and misfit instead of starting from previous inversion results.
    constraints: bool, defaults to false, 
        input to anomalies():choose whether the misfit is calculate everywhere, or interpolated from constraints points
        multiples 
    misfit_sq_tolerance: float,
    delta_misfit_squared_tolerance: float,
    Max_Iterations: int:
    deriv_type: strl defaults to 'annulus', choose method for calculating derivative, can be 'prisms'ArithmeticError
    max_layer_change_per_iter: float; max amount each prism can change per iteration.
    """
    # include_forward_layers = layers_list[~layers_list.isin(exclude_layers)]
    include_forward_layers = pd.Series([k for k, v in layers.items() if k not in exclude_layers])
    
    # if reset is True:
    #     grids_to_prism_layers(layers, plot=False)
    #     forward = forward_grav_layers(
    #         layers, input_grav, exclude_layers, inversion_region, grav_spacing, plot=True, plot_type='xarray')

    spacing = layers[active_layer]['spacing'] 
    misfit_squared_updated=np.Inf  # positive infinity
    delta_misfit_squared=np.Inf  # positive infinity
    ind = include_forward_layers[include_forward_layers == active_layer].index[0]
    ITER=0
    # while delta_misfit_squared (inf) is greater than 1 + least squares tolerance (0.02)
    while delta_misfit_squared > 1+delta_misfit_squared_tolerance:
        ITER+=1 
        print(f"##################################\niteration {ITER}")
        # if ITER==1:
        #     show=True
        #     grav = forward
        # else:
        #     show=False
        #     grav = df_inversion

        # df_anomalies = anomalies(layers, grav, input_grav_column, 
        #     corrections=exclude_layers, regional_method=regional_method, 
        #     inversion_region=inversion_region, grav_spacing=grav_spacing,
        #     filter=filter, trend_order=trend_order, plot=show, plot_type='xarray',)
        if ITER==1:
            pass
        else:
            input_grav['res'] = input_grav[f"iter_{ITER-1}_final_misfit"]
        
        initial_RMS = round(np.sqrt((input_grav.res**2).mean(skipna=True)),2)
        print(f"initial RMS residual = {initial_RMS}mGal")

        # get prisms' coordinates from active layer and layer above
        prisms = layers[active_layer]['prisms'].to_dataframe().reset_index().dropna()
        prisms_above = layers[include_forward_layers[ind-1]]['prisms'].to_dataframe().reset_index().dropna()

        # calculate jacobian
        if deriv_type == 'annulus':
            jac = jacobian_annular(input_grav, input_grav_column, prisms, spacing)
        elif deriv_type == 'prisms':
            jac = jacobian_prism(input_grav, input_grav_column, layers[active_layer]['prisms'], 1, "g_z")
        else:
            print('not valid derivative type')  
        print('finished jacobian')
        # Calculate shift to prism's tops to minimize misfit
        # gives the amount that each column's Z1 needs to change by to have the smallest misfit
        # finds the least-squares solution to jacobian and Grav_Misfit, assigns the first value to Surface_correction
        Surface_correction=lsqr(jac, input_grav.res.values, show=False)[0] 
        for i in range(0,len(prisms)):
            if Surface_correction[i] > max_layer_change_per_iter:
                Surface_correction[i]=max_layer_change_per_iter
            elif Surface_correction[i] < -max_layer_change_per_iter:
                Surface_correction[i]=-max_layer_change_per_iter
        print('finished least squares')
        prisms['correction']=Surface_correction
        prisms_above['correction']=Surface_correction
        print(f"average layers correction {round(np.sqrt((Surface_correction**2).mean()),2)}m")
        # apply above surface corrections 
        if constraints==True:
            prisms['constraints'] = constraints_grid.to_dataframe().reset_index().z
            prisms_above['constraints'] = constraints_grid.to_dataframe().reset_index().z
            prisms['correction'] = prisms.constraints * prisms.correction
            prisms_above['correction'] = prisms_above.constraints * prisms_above.correction
        else:
            print('constraints not applied')

        if ITER==1:
            iter_corrections = prisms.rename(columns={'easting':'x', 'northing':'y'}).copy()
        iter_corrections[f'iter_{ITER}_inital_top'] = prisms.top.copy()

        prisms.top += prisms.correction
        prisms_above.bottom += prisms_above.correction

        iter_corrections[f'iter_{ITER}_final_top'] = prisms.top.copy()

        # apply the z correction to the active prism layer and the above layer with Harmonica 
        # not sure why this doesn't work with .to_xarray(), only with xyz2grid
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
        
        input_grav[f'iter_{ITER}_initial_misfit'] = input_grav.res


        iter_corrections[f"iter_{ITER}_correction"] = prisms.correction.copy()
        # iter_corr = prisms.rename(columns={'easting':'x', 'northing':'y'}).set_index(['x','y']).to_xarray().correction

        print('calculating updated forward gravity')
        input_grav[f"iter_{ITER}_{active_layer}_forward_grav"] = layers[active_layer]['prisms'].prism_layer.gravity(
            coordinates=(input_grav.x, input_grav.y, input_grav.z),
            field = 'g_z')
        
        input_grav[f"iter_{ITER}_{include_forward_layers[ind-1]}_forward_grav"] = layers[include_forward_layers[ind-1]]['prisms'].prism_layer.gravity(
            coordinates=(input_grav.x, input_grav.y, input_grav.z),
            field = 'g_z')
        # input_grav[include_forward_layers[ind-1]]=df_grav.set_index(['x','y']).to_xarray()[include_forward_layers[ind-1]] 
        
        # grav_layers_list = [f"{i}_forward_grav" for i in include_forward_layers]
        # input_grav[f'iter_{ITER}_forward_total']=input_grav[grav_layers_list].sum(axis=1, skipna=True)

        # add updated layers' column names to list
        updated_layers = [active_layer, include_forward_layers[ind-1]]
        updated_layers_list = [f"iter_{ITER}_{i}_forward_grav" for i in updated_layers]
        # add unchanged layers (excluding corrections layers) column names to list
        unchanged_layers = include_forward_layers[~include_forward_layers.str.contains(f"{include_forward_layers[ind-1]}|{active_layer}")]
        unchanged_layers_list = [f"{i}_forward_grav" for i in unchanged_layers]
        # combined list of column names
        updated_forward = updated_layers_list + unchanged_layers_list 
        # recalculate forward gravity with dataframe column names
        input_grav[f'iter_{ITER}_forward_total']=input_grav[updated_forward].sum(axis=1, skipna=True)

        # update the misfit grid
        print('updating the misfits')
        input_grav[f"iter_{ITER}_final_misfit"] = anomalies(layers, input_grav, input_grav_column, 
            input_forward_column = f'iter_{ITER}_forward_total',
            corrections=exclude_layers, regional_method=regional_method, 
            inversion_region=inversion_region, grav_spacing=grav_spacing,
            filter=filter, trend_order=trend_order, plot=False).res
        # input_grav['inv_misfit'] = df_grav.set_index(['x','y']).to_xarray().res 

        final_RMS = round(np.sqrt((input_grav[f"iter_{ITER}_final_misfit"] **2).mean(skipna=True).item()),2)
        print(f"final RMS residual = {final_RMS}mGal")
        # for first iteration, divide infinity by mean square of gravity residuals, inversion will stop once this gets to delta_misfit_squared_tolerance (0.02)
        misfit_sq=(input_grav[f"iter_{ITER}_final_misfit"]**2).mean(skipna=True).item()
        delta_misfit_squared=misfit_squared_updated/misfit_sq
        misfit_squared_updated=misfit_sq # updated 

        layers[active_layer]['inv_grid']=prisms.rename(columns={'easting':'x', 'northing':'y'}).set_index(
                ['y','x']).to_xarray().top
        # layers[active_layer]['inv_grid']=pygmt.xyz2grd(prisms.rename(columns={'easting':'x', 'northing':'y'})[['x','y','top']], region=buffer_reg, registration='p', spacing=spacing)
        
        active_layer_total_difference = layers[active_layer]['inv_grid'] - layers[active_layer]['grid']

        if ITER == Max_Iterations:
            print(f"Inversion terminated after {ITER} iterations with least-squares norm={int(misfit_sq)} because maximum number of iterations ({Max_Iterations}) reached")
            break
        if misfit_sq < misfit_sq_tolerance:
            print(f"Inversion terminated after {ITER} iterations with least-squares norm={int(misfit_sq)} because least-squares norm < {misfit_sq_tolerance}")
            break
      
    # end of inversion iteration WHILE loop
    if delta_misfit_squared < 1+delta_misfit_squared_tolerance:
        print("terminated - no significant variation in least-squares norm ")
    return iter_corrections, input_grav