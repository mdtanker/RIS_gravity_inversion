# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:01:34 2020

@author: matthewt

Use this to covert Geosoft exported grid.xyz into format used for gravity inversion script
to prepare grid in Geosoft, save as a GDB, then add channels for density and control points, then export as XYZ
"""

import pandas as pd

input_file = 'E:/VIC_ROSETTA/bedmap2/bedmap2_surface_wgs84_filled.xyz'   #  <------ put .xyz file name here
output_file = 'C:/Grav_Bathy_Inversion_Code/bedmap2_surface_wgs84_filled.csv'   #  <------ put output file name here

RHO = 0.92
cp = 1

df = pd.read_csv(input_file, header=0, index_col=None, sep='\s+', names=('x','y','z') )

print('import done')

df['rho'] = RHO

df['cp'] = cp

df.to_csv(output_file, sep=',', header=0, index=0)

print('file saved')
