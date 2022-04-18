# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:22:10 2020

@author: matthewt
"""

import sys
sys.path.append('.scripts')
import numpy as np
import matplotlib as mpl
import statistics as stat
from matplotlib import pyplot as plt
from matplotlib import axes
import csv
import pandas as pd
import scipy as sp
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import mpl_toolkits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import Rbf
from scipy.interpolate import RectBivariateSpline
import matplotlib.gridspec as gridspec
from numpy import linalg
from numpy.linalg import lstsq
from scipy.sparse.linalg import lsqr


import scripts.gravbox_with_Nagy_variables
from scripts.gravbox import gravbox
from scripts.grav_column_der import grav_column_der
from scripts.basemap_interp import Interp

import warnings
from datetime import date
from scipy import stats 


FILEGRAVITY = 'data/sample_RIS_gravity_10000m.XYZ'
INPUT_GRAVITY=pd.read_csv(FILEGRAVITY, header=None, index_col=None, sep=',', names=('Y','X','Z','FA','FACP') )
XG=np.array(INPUT_GRAVITY.X) # 2nd col, Northing = X (geophysics convention?)
YG=np.array(INPUT_GRAVITY.Y) # 1st col, Easting = Y (geophysics convention?)
ZG= - np.array(INPUT_GRAVITY.Z) # 3rd col, survey elevation
FA=np.array(INPUT_GRAVITY.FA) # 4th col, Free air grav gridded
FACP=np.array(INPUT_GRAVITY.FACP) #5th col, Free air grav gridded with only bathy control points
NG=len(XG)      #<---- gives size of input data file
print("Successfully Loaded!")
RESG=10000

FILEBATHYMETRY = 'data/sample_RIS_Bathymetry_10000m.XYZ'
INPUT_BATHYMETRY=pd.read_csv(FILEBATHYMETRY, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
XBath=np.array(INPUT_BATHYMETRY.X) 
YBath=np.array(INPUT_BATHYMETRY.Y) 
ZBath= - np.array(INPUT_BATHYMETRY.Z) 
RHOBath=np.array(INPUT_BATHYMETRY.RHO) 
CPBath=np.array(INPUT_BATHYMETRY.CP) 
NBath=len(XBath)      
print("Successfully Loaded!")
RESBATH=10000

FILEBASEMENT = 'data/sample_RIS_Basement_20000m.XYZ'
INPUT_BASEMENT=pd.read_csv(FILEBASEMENT, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
XBase=np.array(INPUT_BASEMENT.X) 
YBase=np.array(INPUT_BASEMENT.Y) 
ZBase=-np.array(INPUT_BASEMENT.Z) 
RHOBase=np.array(INPUT_BASEMENT.RHO) 
CPBase=np.array(INPUT_BASEMENT.CP) 
NBase=len(XBase) 
print("Successfully Loaded!")
RESBASE=10000

FA_Bath=np.zeros(NG)
FA_Base=np.zeros(NG)


Z_Lowest = np.full_like(ZBase, 15000)
for i in range(0,NG):
    # going through gravity data point by point, for each point sum up the result of the function: gravbox_with_Nagy_variables
    FA_Bath[i]=np.sum(gravbox(XG[i], YG[i], ZG[i], XBath-0.5*RESBATH, XBath+0.5*RESBATH,YBath-0.5*RESBATH, YBath+0.5*RESBATH, ZBath, Z_Lower, RHOBath))
    FA_Base[i]=np.sum(gravbox(XG[i], YG[i], ZG[i], XBase-0.5*RESBASE, XBase+0.5*RESBASE,YBase-0.5*RESBASE, YBase+0.5*RESBASE, ZBase, Z_Lowest, RHOBase))
    FA_Tot = FA_Bath
   
FA_Misfit = FA - FA_Tot

FA_Regional=np.full_like(FA, np.nanmean(FA_Misfit))

FA_Residual = FA - FA_Regional


    
for i in range(0,2):
    title = ('Bathymetry Depth', 'Obseved Gravity Anomaly')
    label = ('(meters)', '(mGals)')
    ax = ('ax1', 'ax2')
    ax = list(ax)
    cax = ('cax1', 'cax2')
    cax = list(cax)
    img = ('img1', 'img2')
    img = list(img)
    X = (XBath, XG)
    Y = (YBath,  YG)
    Z = (-ZBath, FA)
    RES = (RESBATH, RESG)

    X_range=np.arange(min(X[i]),max(X[i])+0.0001,RES[i])
    Y_range=np.arange(min(Y[i]),max(Y[i])+0.0001,RES[i])
    East_grid, North_grid = np.meshgrid(Y_range,X_range)

    extent = x_min, x_max, y_min, y_max= [min(Y[i]), max(Y[i]), min(X[i]), max(X[i])]
    grid_Z = griddata((Y[i], X[i]), Z[i], (East_grid, North_grid), method='linear')
    
    fig1 = plt.figure(1, (10,10))
    ax[i] = fig1.add_subplot(1,2,i+1, adjustable='box', aspect=1)
   
    img[i] = ax[i].contourf(grid_Z, 100, cmap='jet', extent=extent)
    
    ax[i].set_title(title[i])
    ax[i].set_xlabel('Easting (Km)')
    ax[i].set_ylabel('Northing (Km)')
    
    ax[i].set_ylim(-1385000,-485000)
    ax[i].set_xlim(-555000,345000)
    ax[i].set_yticks( ticks = np.arange(min(XBath)+30000, max(XBath)+30000, 100000))
    ax[i].set_yticklabels(labels = np.arange(int(0.001*(min(XBath)+30000)), int(0.001*(max(XBath)+30000)), int(100)))
    ax[i].set_xticks( ticks = np.arange(min(YBath+40000), max(YBath)+40000, 200000))
    ax[i].set_xticklabels(labels = np.arange(int(0.001*(min(YBath)+40000)), int(0.001*(max(YBath)+40000)), int(200)))
    #ax[i].margins(2,2)
    
    divider  = make_axes_locatable(ax[i])
    cax[i] = divider.append_axes('right', size='5%', pad = 0.1)
    cb1 = fig1.colorbar(img[i], label=label[i], cax=cax[i])
    cb1.set_ticks( ticks = np.arange(min(Z[i]), max(Z[i]), (abs(max(Z[i])-min(Z[i])))/10))
    
plt.suptitle('Input Geologic Model Layers', fontsize='xx-large', y=1.02) 
plt.tight_layout()
plt.show()

for i in range(0,6):
    title = ('Observed Gravity Anomaly', 'Gravity Anomaly from Control Points', 'Regional Gravity Anomaly', 'Residual Gravity Anomaly', 'Forward Gravity Anomaly', 'Gravity Misfit')
    ax = ('ax1', 'ax2', 'ax3', 'ax4', 'ax5', 'ax6')
    ax = list(ax)
    cax = ('cax1', 'cax2', 'cax3', 'cax4', 'cax5', 'cax6')
    cax = list(cax)
    img = ('img1', 'img2', 'img3', 'img4', 'img5', 'img6')
    img = list(img)

    Z = (FA, FACP, FA_Regional, FA_Residual, FA_Tot, FA_Misfit)

    XG_range=np.arange(min(XG),max(XG)+0.0001,RESG)
    YG_range=np.arange(min(YG),max(YG)+0.0001,RESG)
    East_gridG, North_gridG = np.meshgrid(YG_range,XG_range)
    extentG = xG_min, xG_max, yG_min, yG_max = [min(YG), max(YG), min(XG), max(XG)]
    
    grid_Z = griddata((YG, XG), Z[i], (East_gridG, North_gridG), method='linear')

    fig2 = plt.figure(2, (10,8))
    ax[i] = fig2.add_subplot(3,2,i+1, adjustable='box', aspect=1)
    img[i] = ax[i].contourf(grid_Z, 100, cmap='jet', extent=extentG)
    ax[i].set_title(title[i])
    ax[i].set_xlabel('Easting (Km)')
    ax[i].set_ylabel('Northing (Km)')
    ax[i].set_ylim(-1340000,-540000)
    ax[i].set_xlim(-495000,305000)
    ax[i].set_yticks( ticks = np.arange(min(XBath)+30000, max(XBath)+30000, 100000))
    ax[i].set_yticklabels(labels = np.arange(int(0.001*(min(XBath)+30000)), int(0.001*(max(XBath)+30000)), int(100)))
    ax[i].set_xticks( ticks = np.arange(min(YBath+40000), max(YBath)+40000, 200000))
    ax[i].set_xticklabels(labels = np.arange(int(0.001*(min(YBath)+40000)), int(0.001*(max(YBath)+40000)), int(200)))

    divider  = make_axes_locatable(ax[i])
    cax[i] = divider.append_axes('right', size='5%', pad = 0.05)
    cb2 = fig2.colorbar(img[i], label='(mGals)', cax=cax[i])

#plt.subplots_adjust(left=2, bottom=2, right=20, top=20)
plt.suptitle('Gravity Grids', fontsize='xx-large', y=1.02)   
plt.tight_layout()
plt.show()
print('Done Plotting Gravity Grids\n')
