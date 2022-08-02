# RIS_gravity_inversion
Here we use gravity data from Antarctica's Ross Ice Shelf region, mainly airborne data from the ROSETTA-Ice project, to perform a gravity inverison to model the sub ice shelf bathymetry. 

## To Do

### Issues
* test with layers with NaN's (instead of ice and water elev = 0 for no ice, make nans)
* fix constraints on GL plotting
### Short-term
* add spacing parameter to grdfilters
* use rosetta ice thickness
* do forward model with rosetta density
* use Boule to remove normal gravity
    - using rosetta "FAG_levelled" channel
* fix density inversion
* use rioxarray.pad_box or xarray.pad for padding
* use rioxarray.reproject_match for resampling
* use vd.median_distance for constraints grid construction
* use hm.EquivalentSources.jacobian to calculate jac.
* add hm.EquivalentSources as method for regional calc
* use vd.Trend.jacobian for regularization
* use vd.base.least_squares for optimization

### Long-term
* fix pyvista to work on remote server
    * use pvxarray for plotting topographies in 3D
* use discretize package to increase grid spacing outside of inv_reg
* implement vertically increasing density to both ice and sediment (like harmonica.tesseroid_gravity)

## Questions:
* should we use raw gravity observation points, or interpolated grid?

## Other Gravity Inversion Software 

### Geosoft Oasis Montaj 
* 3D 
* Square grid network 

### MiraGeoscience 
* 3D
* offer Vertical Prisms inversion

### Fatiando a Terra / Harmonica 

### SimPEG 
* 3D
* mesh based, not well suited (yet) for vertical prism inversion

### PyGIMLI 
* 3D
* mesh based

### GNS Woodward Fortran Code 
* 3D 
* Triangular grid network 
* Irregular grid spacing 
* Accepts constraint cells 

### GNS Nagy / Woodward Python Code 
* 3D 
* Same optimization of GNS fortran code 
* Square grid network 
* different grid spacing for each layer 

### Growth 3.0 Fortran Code
* 3D
* accepts seeds

 

 

 

 
