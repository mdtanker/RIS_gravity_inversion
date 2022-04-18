# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:01:01 2020

@author: matthewt
"""

# import python packages
import sys
import numpy as np
import matplotlib as mpl
import statistics as stat
from matplotlib import pyplot as plt
from matplotlib import axes
import csv
import pandas as pd
import scipy as sp
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
import mpl_toolkits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import Rbf
import matplotlib.gridspec as gridspec
from numpy import linalg
from numpy.linalg import lstsq
from scipy.sparse.linalg import lsqr
from gravbox_with_Nagy_variables import gravbox_with_Nagy_variables
import warnings
from datetime import date
from grav_column_der import grav_column_der
from scipy import stats 

# import GUI program (QtDesign) packages
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTextEdit

#import .py geometry file from Qt Designer
import grav_bathy_inversion_GUI_geometry    #<---- add .py geometry file

from scipy.ndimage import filters


FILEGRAVITY = 'sample_RIS_gravity_10000m.XYZ'
INPUT_GRAVITY=pd.read_csv(FILEGRAVITY, header=None, index_col=None, sep=',', names=('Y','X','Z','FA','FACP') )
XG=np.array(INPUT_GRAVITY.X) # 2nd col, Northing = X (geophysics convention?)
YG=np.array(INPUT_GRAVITY.Y) # 1st col, Easting = Y (geophysics convention?)
ZG=-np.array(INPUT_GRAVITY.Z) # 3rd col, survey elevation
FA=np.array(INPUT_GRAVITY.FA) # 4th col, Free air grav gridded
FACP=np.array(INPUT_GRAVITY.FACP) #5th col, Free air grav gridded with only bathy control points
NG=len(XG)      #<---- gives size of input data file
print("Successfully Loaded!")


FILEICE = 'sample_RIS_Water_10000m.XYZ'
INPUT_ICE=pd.read_csv(FILEICE, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
XI=np.array(INPUT_ICE.X) 
YI=np.array(INPUT_ICE.Y) 
ZI=-np.array(INPUT_ICE.Z) 
RHOI=np.array(INPUT_ICE.RHO) 
CPI=np.array(INPUT_ICE.CP) 
NI=len(XI)
print("Successfully Loaded!")


FILEWATER = 'sample_RIS_Water_10000m.XYZ'
INPUT_WATER=pd.read_csv(FILEWATER, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
XW=np.array(INPUT_WATER.X) 
YW=np.array(INPUT_WATER.Y) 
ZW=-np.array(INPUT_WATER.Z) 
RHOW=np.array(INPUT_WATER.RHO) 
CPW=np.array(INPUT_WATER.CP) 
NW=len(XW)
print("Successfully Loaded!")


FILEBATHYMETRY = 'sample_RIS_Bathymetry_10000m.XYZ'
INPUT_BATHYMETRY=pd.read_csv(FILEBATHYMETRY, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
XBath=np.array(INPUT_BATHYMETRY.X) 
YBath=np.array(INPUT_BATHYMETRY.Y) 
ZBath=-np.array(INPUT_BATHYMETRY.Z) 
RHOBath=np.array(INPUT_BATHYMETRY.RHO) 
CPBath=np.array(INPUT_BATHYMETRY.CP) 
NBath=len(XBath)      
print("Successfully Loaded!")


FILEBASEMENT = 'sample_RIS_Basement_20000m.XYZ'
INPUT_BASEMENT=pd.read_csv(FILEBASEMENT, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
XBase=np.array(INPUT_BASEMENT.X) 
YBase=np.array(INPUT_BASEMENT.Y) 
ZBase=-np.array(INPUT_BASEMENT.Z) 
RHOBase=np.array(INPUT_BASEMENT.RHO) 
CPBase=np.array(INPUT_BASEMENT.CP) 
NBase=len(XBase) 
print("Successfully Loaded!")

"""
FILEMOHO = 'sample_RIS_Moho_20000m.XYZ'
INPUT_MOHO=pd.read_csv(FILEMOHO, header=None, index_col=None, sep=',', names=('Y','X','Z','RHO','CP') )
XM=np.array(INPUT_MOHO.X) 
YM=np.array(INPUT_MOHO.Y) 
ZM=-np.array(INPUT_MOHO.Z) 
RHOM=np.array(INPUT_MOHO.RHO) 
CPM=np.array(INPUT_MOHO.CP) 
NM=len(XM) 
print("Successfully Loaded!")
"""


RESG=10000
RESI=10000
RESW=10000
RESBATH=10000
RESBASE=10000

#RESM=10000
       
NCPBath = int(np.sum(CPBath))
NCPBase = int(np.sum(CPBase))
#NCPM = int(np.sum(CPM))

# coords of control points = array of zeros of length total number of control points
XCBath = np.zeros(NCPBath)
YCBath = np.zeros(NCPBath)
XCBase = np.zeros(NCPBase)
YCBase = np.zeros(NCPBase)
#XCM = np.zeros(NCPM)
#YCM = np.zeros(NCPM)
 
# index of layer = array of zeros of the length of the file
INDEXBath=np.zeros(NBath)
INDEXBase=np.zeros(NBase)
#INDEXM=np.zeros(NM)

         
ICPBath=-1
for i in range(0,NBath):          # iterate over every point in Bath
    if CPBath[i] > 0.5:           # if the control point value is >.5
        ICPBath=ICPBath+1             #
        XCBath[ICPBath]=XBath[i]      # the X and Y coords are added to the respective index of XCBath 
        YCBath[ICPBath]=YBath[i]     
    INDEXBath[i]=i

ICPBase=-1
for i in range(0,NBase):
    if CPBase[i] > 0.5:
        ICPBase=ICPBase+1
        XCBase[ICPBase]=XBase[i]
        YCBase[ICPBase]=YBase[i]
    INDEXBase[i]=i
"""   
ICPM=-1
for i in range(0,NM):
    if CPM[i] > 0.5:
        ICPM=ICPM+1
        XCM[ICPM]=XM[i]
        YCM[ICPM]=YM[i]
    INDEXM[i]=i
"""
# Gridding / Plotting Input Layers
# plotting in a for-loop

"""
plt.close('all')
for i in range(0,6):
    title = ('Ice Surface Height', 'Water Surface Height', 'Bathymetry Depth', 'Basement Depth', 'Moho Depth', 'Obseved Gravity Anomaly')
    label = ('(meters)', '(meters)', '(meters)', '(meters)', '(meters)', '(mGals)')
    ax = ('ax1', 'ax2', 'ax3', 'ax4', 'ax5', 'ax6')
    ax = list(ax)
    cax = ('cax1', 'cax2', 'cax3', 'cax4', 'cax5', 'cax6')
    cax = list(cax)
    img = ('img1', 'img2', 'img3', 'img4', 'img5', 'img6')
    img = list(img)
    X = (XI, XW, XBath, XBase, XM, XG)
    Y = (YI, YW, YBath, YBase, YM, YG)
    Z = (ZI, ZW, ZBath, ZBase, ZM, FA)
    RES = (RESI, RESW, RESBATH, RESBASE, RESM, RESG)
    XC = (XCBath, XCBase, XCM)
    YC = (YCBath, YCBase, YCM)

    X_range=np.arange(min(X[i]),max(X[i])+0.0001,RES[i])
    Y_range=np.arange(min(Y[i]),max(Y[i])+0.0001,RES[i])
    East_grid, North_grid = np.meshgrid(Y_range,X_range)

    extent = x_min, x_max, y_min, y_max= [min(Y[i]), max(Y[i]), min(X[i]), max(X[i])]
    grid_Z = griddata((Y[i], X[i]), Z[i], (East_grid, North_grid), method='linear')
    
    fig1 = plt.figure(1, (10,10))
    ax[i] = fig1.add_subplot(3,2,i+1, adjustable='box', aspect=1)
   
    img[i] = ax[i].contourf(grid_Z, 100, cmap='jet', extent=extent)
    
    if i > 1 and i < 5:
        ax[i].plot(YC[i-2], XC[i-2], 'ko', markersize=2)
    else:
        pass
    
    ax[i].set_title(title[i])
    ax[i].set_xlabel('Easting (Km)')
    ax[i].set_ylabel('Northing (Km)')
    
    ax[i].set_ylim(-1385000,-485000)
    ax[i].set_xlim(-555000,345000)
    ax[i].set_yticks( ticks = np.arange(min(XI)+30000, max(XI)+30000, 100000))
    ax[i].set_yticklabels(labels = np.arange(int(0.001*(min(XI)+30000)), int(0.001*(max(XI)+30000)), int(100)))
    ax[i].set_xticks( ticks = np.arange(min(YI+40000), max(YI)+40000, 200000))
    ax[i].set_xticklabels(labels = np.arange(int(0.001*(min(YI)+40000)), int(0.001*(max(YI)+40000)), int(200)))
    #ax[i].margins(2,2)
    
    divider  = make_axes_locatable(ax[i])
    cax[i] = divider.append_axes('right', size='5%', pad = 0.1)
    cb1 = fig1.colorbar(img[i], label=label[i], cax=cax[i])
    cb1.set_ticks( ticks = np.arange(min(Z[i]), max(Z[i]), (abs(max(Z[i])-min(Z[i])))/10))
    
plt.suptitle('Input Geologic Model Layers', fontsize='xx-large', y=1.02) 
plt.tight_layout()
plt.show()
"""

print("Preparing grids")

XG_range=np.arange(min(XG),max(XG)+0.0001,RESG)
YG_range=np.arange(min(YG),max(YG)+0.0001,RESG)
East_gridG, North_gridG = np.meshgrid(YG_range,XG_range)

RHOIGRID=griddata((YI,XI),RHOI,(East_gridG, North_gridG), method='linear')
RHOWGRID=griddata((YW,XW),RHOW,(East_gridG, North_gridG), method='linear')
RHOBathGRID=griddata((YBath,XBath),RHOBath,(East_gridG, North_gridG), method='linear')
RHOBaseGRID=griddata((YBase,XBase),RHOBase,(East_gridG, North_gridG), method='linear')

I_GRID=griddata((YI,XI),-ZI,(East_gridG, North_gridG), method='linear')
W_GRID=griddata((YW,XW),-ZW,(East_gridG, North_gridG), method='linear')
Bath_GRID=griddata((YBath,XBath),ZBath,(East_gridG, North_gridG), method='linear')
Base_GRID=griddata((YBase,XBase),ZBase,(East_gridG, North_gridG), method='linear')
#M_GRID=griddata((YM,XM),ZM,(East_gridG, North_gridG), method='linear')
   
"""
###########################################################################
# building vector of control points
# number of control points in Bathy = sum of all numbers in CPBath (CP=1, non CP=0)
NCPBath = int(np.sum(CPBath))
NCPBase = int(np.sum(CPBase))
NCPM = int(np.sum(CPM))

# coords of control points = array of zeros of length total number of control points
XCBath = np.zeros(NCPBath)
YCBath = np.zeros(NCPBath)
XCBase = np.zeros(NCPBase)
YCBase = np.zeros(NCPBase)
XCM = np.zeros(NCPM)
YCM = np.zeros(NCPM)

# index of layer = array of zeros of the length of the file
INDEXBath=np.zeros(NBath)
INDEXBase=np.zeros(NBase)
INDEXM=np.zeros(NM)

ICPBath=-1
for i in range(0,NBath):          # iterate over every point in Bath
    if CPBath[i] > 0.5:           # if the control point value is >.5
        ICPBath=ICPBath+1             #
        XCBath[ICPBath]=XBath[i]      # the X and Y coords are added to the respective index of XCBath 
        YCBath[ICPBath]=YBath[i]     
    INDEXBath[i]=i
ICPBase=-1
for i in range(0,NBase):
    if CPBase[i] > 0.5:
        ICPBase=ICPBase+1
        XCBase[ICPBase]=XBase[i]
        YCBase[ICPBase]=YBase[i]
    INDEXBase[i]=i 
ICPM=-1
for i in range(0,NM):
    if CPM[i] > 0.5:
        ICPM=ICPM+1
        XCM[ICPM]=XM[i]
        YCM[ICPM]=YM[i]
    INDEXM[i]=i
""" 
    
NGX=XG_range.size
NGY=YG_range.size

XGRIDVEC=np.zeros([NGX*NGY])
YGRIDVEC=np.zeros([NGX*NGY])

I_GRIDVEC=np.zeros([NGX*NGY])
W_GRIDVEC=np.zeros([NGX*NGY])
Bath_GRIDVEC=np.zeros([NGX*NGY])
Base_GRIDVEC=np.zeros([NGX*NGY])
#M_GRIDVEC=np.zeros([NGX*NGY])

ZIGRID=griddata((YI,XI),ZI,(East_gridG, North_gridG), method='linear')
ZWGRID=griddata((YW,XW),ZW,(East_gridG, North_gridG), method='linear')
ZBathGRID=griddata((YBath,XBath),ZBath,(East_gridG, North_gridG), method='linear')
ZBaseGRID=griddata((YBase,XBase),ZBase,(East_gridG, North_gridG), method='linear')

RHOIGRIDVEC=np.zeros([NGX*NGY])
RHOWGRIDVEC=np.zeros([NGX*NGY])
RHOBathGRIDVEC=np.zeros([NGX*NGY])
RHOBaseGRIDVEC=np.zeros([NGX*NGY])
#RHOMGRIDVEC=np.zeros([NGX*NGY])   # ???? not sure if I need this for the last layer (Moho)
ZIGRIDVEC=np.zeros([NGX*NGY])
ZWGRIDVEC=np.zeros([NGX*NGY])
ZBathGRIDVEC=np.zeros([NGX*NGY])
ZBaseGRIDVEC=np.zeros([NGX*NGY])


INDGRID=-1
for ix in range(0,NGX):
    for iy in range(0,NGY):
        INDGRID=INDGRID+1
        XGRIDVEC[INDGRID]=North_gridG[ix,iy]
        YGRIDVEC[INDGRID]=East_gridG[ix,iy]
        RHOIGRIDVEC[INDGRID]=RHOIGRID[ix,iy]
        RHOWGRIDVEC[INDGRID]=RHOWGRID[ix,iy]
        RHOBathGRIDVEC[INDGRID]=RHOBathGRID[ix,iy]
        RHOBaseGRIDVEC[INDGRID]=RHOBaseGRID[ix,iy]
        #RHOMGRIDVEC[INDGRID]=RHOMGRID[ix,iy]     # ???? not sure if I need this for the last layer (Moho)
        I_GRIDVEC[INDGRID]=I_GRID[ix,iy]
        W_GRIDVEC[INDGRID]=W_GRID[ix,iy]
        Bath_GRIDVEC[INDGRID]=-Bath_GRID[ix,iy]
        Base_GRIDVEC[INDGRID]=-Base_GRID[ix,iy]
        #M_GRIDVEC[INDGRID]=-M_GRID[ix,iy]     # ???? not sure if I need this for the last layer (Moho)
        ZIGRIDVEC[INDGRID]=ZIGRID[ix,iy]
        ZWGRIDVEC[INDGRID]=ZWGRID[ix,iy]
        ZBathGRIDVEC[INDGRID]=ZBathGRID[ix,iy]
        ZBaseGRIDVEC[INDGRID]=ZBaseGRID[ix,iy]
       
GRID_POINTS=np.zeros([NGX*NGY,2])
GRID_POINTS[:,0]=YGRIDVEC
GRID_POINTS[:,1]=XGRIDVEC

# ice density on spacing of water
intfunDENSI=LinearNDInterpolator(GRID_POINTS,RHOIGRIDVEC)
RHOI_W=intfunDENSI(YW,XW)
RHOI_W[np.isnan(RHOI_W)]=np.mean(RHOI_W[np.isfinite(RHOI_W)])
# water density on spacing of bath
intfunDENSW=LinearNDInterpolator(GRID_POINTS,RHOWGRIDVEC)
RHOW_Bath=intfunDENSW(YBath,XBath)
RHOW_Bath[np.isnan(RHOW_Bath)]=np.mean(RHOW_Bath[np.isfinite(RHOW_Bath)])
# bathymetry density on spacing of basement
intfunDENSBath=LinearNDInterpolator(GRID_POINTS,RHOBathGRIDVEC)
RHOBath_Base=intfunDENSBath(YBase,XBase)
RHOBath_Base[np.isnan(RHOBath_Base)]=np.mean(RHOBath_Base[np.isfinite(RHOBath_Base)])
# basement density on spacing of moho
#intfunDENSBase=LinearNDInterpolator(GRID_POINTS,RHOBaseGRIDVEC)
#RHOBase_M=intfunDENSBase(YM,XM)
#RHOBase_M[np.isnan(RHOBase_M)]=np.mean(RHOBase_M[np.isfinite(RHOBase_M)])

# reproject Z values onto lower surfaces
# ice Z on spacing of water
intfunZI=LinearNDInterpolator(GRID_POINTS,ZIGRIDVEC)
ZW_wrt_I=intfunZI(YW,XW)
ZW_wrt_I[np.isnan(ZW_wrt_I)]=np.mean(ZW_wrt_I[np.isfinite(ZW_wrt_I)])

# water Z on spacing of bathymetry
intfunZW=LinearNDInterpolator(GRID_POINTS,ZWGRIDVEC)
ZBath_wrt_W=intfunZW(YBath,XBath)
ZBath_wrt_W[np.isnan(ZBath_wrt_W)]=np.mean(ZBath_wrt_W[np.isfinite(ZBath_wrt_W)])

# bathymetry Z on spacing of basement
intfunZBath=LinearNDInterpolator(GRID_POINTS,ZBathGRIDVEC)
ZBase_wrt_Bath=intfunZBath(YBath,XBath)
ZBase_wrt_Bath[np.isnan(ZBase_wrt_Bath)]=np.mean(ZBase_wrt_Bath[np.isfinite(ZBase_wrt_Bath)])

# basementy Z on spacing of flat grid at 12000 m
XLowest = XBase
YLowest = YBase
ZLowest = np.zeros(NBase)
ZLowest = 12000
 
intfunZBase=LinearNDInterpolator(GRID_POINTS,ZBaseGRIDVEC)
ZLowest_wrt_Base=intfunZBase(YLowest,XLowest)
ZLowest_wrt_Base[np.isnan(ZLowest_wrt_Base)]=np.mean(ZLowest_wrt_Base[np.isfinite(ZLowest_wrt_Base)])


#############################################
# ???????????????????????
# calculate regional gravity field from control points
print("Calculating regional gravity field from control points \n")
    # creating new arrays full of zeros of length NG
FA_I=np.zeros(NG)
FA_W=np.zeros(NG)
FA_Bath=np.zeros(NG)
FA_Base=np.zeros(NG)
#FA_CPM=np.zeros(NG)
FA_Tot=np.zeros(NG)


Z2I=0*ZI
Z2W=0*ZW
Z2Bath=0*ZBath
Z2Base=0*ZBase
#Z2M=0*ZM

for i in range(0,NG):
    # going through gravity data point by point, for each point sum up the result of the function: gravbox_with_Nagy_variables
    FA_I[i]=np.sum(gravbox_with_Nagy_variables(     XG[i], # XG 
                                                    YG[i], # YG
                                                    ZG[i], # ZG
                                                    XI,    # XLayer
                                                    YI,    # YLayer
                                                    ZI,    # ZLayer
                                                    ZW_wrt_I,
                                                    RESI,  # RESLayer
                                                    RHOI))   # rho
    FA_W[i]=np.sum(gravbox_with_Nagy_variables(XG[i], YG[i], ZG[i], XW, YW, ZW, ZBath_wrt_W, RESW,  RHOW))
    FA_Bath[i]=np.sum(gravbox_with_Nagy_variables(XG[i], YG[i], ZG[i], XBath, YBath, ZBath, ZBase_wrt_Bath, RESBATH,  RHOBath))
    FA_Base[i]=np.sum(gravbox_with_Nagy_variables(XG[i], YG[i], ZG[i], XBase, YBase, ZBase, ZLowest_wrt_Base, RESBASE,  RHOBase))
    #FA_CPM[i]=np.sum(gravbox_with_Nagy_variables(XG[i], YG[i], ZG[i], XM, YM, ZM, RESM,  RHOM))

    FA_Tot = - FA_I - FA_W + FA_Bath + FA_Base #+ FA_CPM
    
FA_DCP = FACP - FA_Tot

print("Regional Trend",np.nanmean(FA_DCP))

TREND_TYPE="Constant value"
#TREND_TYPE=="regional"

if TREND_TYPE=="Constant value":
    FA_Regional=np.nanmean(FA_DCP)*np.ones([NG])    # changed mp.mean to np.nanmean to fix issue of not griding Regional or Residual when layers had different cell sizes
else:
    ATEMP=np.lib.column_stack((np.ones(XG.size), XG, YG))
    C,RESID,RANK,SIGMA=linalg.lstsq(ATEMP,FA_DCP)
    FA_Regional=C[0]*np.ones([NG])+C[1]*XG+C[2]*YG

FA_Residual = FA - FA_Regional
FA_Misfit = FA - FA_Tot
#################################################
# Plotting Gravity Grids

# plotting in a for-loop
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
    ax[i].set_yticks( ticks = np.arange(min(XI)+30000, max(XI)+30000, 100000))
    ax[i].set_yticklabels(labels = np.arange(int(0.001*(min(XI)+30000)), int(0.001*(max(XI)+30000)), int(100)))
    ax[i].set_xticks( ticks = np.arange(min(YI+40000), max(YI)+40000, 200000))
    ax[i].set_xticklabels(labels = np.arange(int(0.001*(min(YI)+40000)), int(0.001*(max(YI)+40000)), int(200)))

    divider  = make_axes_locatable(ax[i])
    cax[i] = divider.append_axes('right', size='5%', pad = 0.05)
    cb2 = fig2.colorbar(img[i], label='(mGals)', cax=cax[i])
    if i==1:
         cb2.set_ticks( ticks = np.arange(0,1,1))
    else:
        #cb2.set_ticks( ticks = np.arange(min(Z[i]), max(Z[i]), (abs(max(Z[i])-min(Z[i])))/10))
        #cb2.set_ticks( ticks = np.arange(min(Z[i]), max(Z[i]), (abs(max(Z[i])-min(Z[i])))/10))
        pass
#plt.subplots_adjust(left=2, bottom=2, right=20, top=20)
plt.suptitle('Gravity Grids', fontsize='xx-large', y=1.02)   
plt.tight_layout()
plt.show()
print('Done Plotting Gravity Grids\n')
