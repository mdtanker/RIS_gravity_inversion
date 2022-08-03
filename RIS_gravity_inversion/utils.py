import pygmt
from pyproj import Transformer
import pandas as pd
import geopandas as gpd
import xarray as xr
import verde as vd
import rioxarray
import numpy as np
import warnings
from typing import Union
import seaborn as sns
import matplotlib.pyplot as plt

def mask_from_shp(
    shapefile, 
    invert=True, 
    xr_grid=None, 
    grid_file=None, 
    region=None, 
    spacing=None, 
    masked=False, 
    crs='epsg:3031'):
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
        coords = vd.grid_coordinates(region=region, spacing=spacing, 
        pixel_register=True)
        ds = vd.make_xarray_grid(coords, np.ones_like(coords[0]), dims=('y', 'x'), 
        data_names='z')
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
    warnings.filterwarnings('ignore', message="pandas.Int64Index")
    warnings.filterwarnings('ignore', message="pandas.Float64Index")
    
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

def make_grid(
    region: Union[str, np.ndarray],
    spacing: float, 
    value: float, 
    name: str,
    ):
    """
    Return a xr.Dataset with variable 'name' of constant value.

    Parameters
    ----------
    region : str or np.ndarray
        GMT format region for the inverion, by default is extent of gravity data, 
    spacing : float
        spacing for grid
    value : float
        constant value to use for variable
    name : str
        name for variable
    """
    coords = vd.grid_coordinates(region=region, spacing=spacing, pixel_register=True)
    data = np.ones_like(coords[0])+value
    grid = vd.make_xarray_grid(coords, data, dims=['y','x'], data_names=name)
    return grid

def raps(
    data : Union[pd.DataFrame, xr.DataArray, xr.Dataset], 
    names : np.ndarray,
    plot_type: str = 'mpl',
    filter: str = None,
    **kwargs
    ):
    """
    Compute and plot the Radially Averaged Power Spectrum input data.

    Parameters
    ----------
    data : Union[pd.DataFrame, str, list, xr.Dataset, xr.Dataarray]
        if dataframe: need with columns 'x', 'y', and other columns to calc RAPS for.
        if str: should be a .nc or .tif file.
        if list: list of grids or filenames.
    names : np.ndarray
        names of pd.dataframe columns, xr.dataset variables, xr.dataarray variable, or files to calculate and plot RAPS for.
    plot_type : str, optional
        choose whether to plot with PyGMT or matplotlib, by default 'mpl'
    filter : str
        GMT string to use for pre-filtering data, ex. "c100e3+h" is a 100km low-pass cosine filter, by default is None.
    Keyword Args
    ------------
    region : Union[str, np.ndarray]
        grid region if input is not a grid 
    spacing : float
        grid spacing if input is not a grid
    """
    region = kwargs.get('region', None)
    spacing = kwargs.get('spacing', None)

    if plot_type == 'pygmt':
            import random
            spec = pygmt.Figure()
            spec.basemap(
                    region='10/1000/.001/10000', 
                    projection="X-10cl/10cl", 
                    frame=["WSne", 'xa1f3p+l"Wavelength (km)"',
                        'ya1f3p+l"Power (mGal@+2@+km)"'])
    elif plot_type == 'mpl':
        plt.figure()
    for i,j in enumerate(names):
            if isinstance(data, pd.DataFrame):
                df = data
                grid = pygmt.xyz2grd(df[['x','y', j]], registration='p', 
                            region=region, spacing=spacing,)
                pygmt.grdfill(grid, mode='n', outgrid='tmp_outputs/fft.nc')
                grid = 'tmp_outputs/fft.nc'
            elif isinstance(data, str):
                grid = data
            elif isinstance(data, list):
                data[i].to_netcdf('tmp_outputs/fft.nc')
                grid = 'tmp_outputs/fft.nc'
            elif isinstance(data, xr.Dataset):
                data[j].to_netcdf('tmp_outputs/fft.nc')
                grid = 'tmp_outputs/fft.nc'
            elif isinstance(data, xr.DataArray):
                data.to_netcdf('tmp_outputs/fft.nc')
                grid = 'tmp_outputs/fft.nc'
            if filter is not None:
                with pygmt.clib.Session() as session:
                    fin = grid
                    fout = 'tmp_outputs/fft.nc'
                    args = f"{fin} -F{filter} -D0 -G{fout}"
                    session.call_module('grdfilter', args)
                grid = 'tmp_outputs/fft.nc'
            with pygmt.clib.Session() as session:
                fin = grid 
                fout = 'tmp_outputs/raps.txt'
                args = f"{fin} -Er+wk -Na+d -G{fout}"
                session.call_module('grdfft', args)
            if plot_type == 'mpl':
                raps = pd.read_csv('tmp_outputs/raps.txt', header=None, 
                    delimiter='\t', names=('Wavelength (km)','Radially Averaged Power','stdev'))
                ax = sns.lineplot(raps['Wavelength (km)'], raps['Radially Averaged Power'], label=j, 
                    palette='viridis')
                ax = sns.scatterplot(x=raps['Wavelength (km)'], y=raps['Radially Averaged Power'])
            elif plot_type == 'pygmt':
                color=f"{random.randrange(255)}/{random.randrange(255)}/{random.randrange(255)}"
                spec.plot('tmp_outputs/raps.txt', pen=f"1p,{color}")
                spec.plot(
                        'tmp_outputs/raps.txt',
                        color=color,
                        style='T5p',
                        # error_bar='y+p0.5p',
                        label=j,
                        )
    if plot_type == 'mpl':
            ax.invert_xaxis()
            ax.set_yscale('log')
            ax.set_xlim(200,0)
            # ax.set_xscale('log')
    elif plot_type == 'pygmt':
            spec.show()
    
    # plt.phase_spectrum(df_anomalies.ice_forward_grav, label='phase spectrum')
    # plt.psd(df_anomalies.ice_forward_grav, label='psd')
    # plt.legend()

def coherency(
    grids : list, 
    label : str,
    **kwargs
    ):
    """
    Compute and plot the Radially Averaged Power Spectrum input data.

    Parameters
    ----------
    grids : list
        list of 2 grids to calculate the cohereny between.
        grid format can be str (filename), xr.DataArray, or pd.DataFrame.
    label : str
        used to label line.
    Keyword Args
    ------------
    region : Union[str, np.ndarray]
        grid region if input is pd.DataFrame
    spacing : float
        grid spacing if input is pd.DataFrame
    """
    region = kwargs.get('region', None)
    spacing = kwargs.get('spacing', None)

    # plt.figure()

    if isinstance(grids[0], (str, xr.DataArray)):
        pygmt.grdfill(grids[0], mode='n', outgrid=f'tmp_outputs/fft_1.nc')
        pygmt.grdfill(grids[1], mode='n', outgrid=f'tmp_outputs/fft_2.nc')
    
    elif isinstance(grids[0], pd.DataFrame):
        grid1 = pygmt.xyz2grd(grids[0], registration='p', 
                    region=region, spacing=spacing,)
        grid2 = pygmt.xyz2grd(grids[1], registration='p', 
                    region=region, spacing=spacing,)    
        pygmt.grdfill(grid1, mode='n', outgrid=f'tmp_outputs/fft_1.nc')
        pygmt.grdfill(grid2, mode='n', outgrid=f'tmp_outputs/fft_2.nc')

    with pygmt.clib.Session() as session:
        fin1 = "tmp_outputs/fft_1.nc"
        fin2 = "tmp_outputs/fft_2.nc"
        fout = 'tmp_outputs/coherency.txt'
        args = f"{fin1} {fin2} -E+wk+n -Na+d -G{fout}"
        session.call_module('grdfft', args)

    df = pd.read_csv('tmp_outputs/coherency.txt', header=None,delimiter='\t', 
    names=(
        'Wavelength (km)', 
        'Xpower', 'stdev_xp',
        'Ypower', 'stdev_yp',
        'coherent power', 'stdev_cp',
        'noise power', 'stdev_np',
        'phase', 'stdev_p',
        'admittance', 'stdev_a',
        'gain', 'stdev_g',
        'coherency', 'stdev_c'))
    ax = sns.lineplot(df['Wavelength (km)'], df.coherency, label=label)
    ax = sns.scatterplot(x=df['Wavelength (km)'], y=df.coherency)
    
    ax.invert_xaxis()
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_xlim(2000, 10)
    # return ax
    """
    Examples:
    utils.coherency(
        grids=[df_anomalies[['x','y','Gobs']], 
            df_anomalies[['x','y','ice_forward_grav']]],
        region=inv_reg,
        spacing=grav_spacing,
        )
    """

