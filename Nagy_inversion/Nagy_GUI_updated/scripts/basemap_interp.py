# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:14:12 2020

@author: matthewt
"""
import numpy as np
def Interp(datain,xin,yin,xout,yout,interpolation='NearestNeighbour'):

    """
       Interpolates a 2D array onto a new grid (only works for linear grids), 
       with the Lat/Lon inputs of the old and new grid. Can perfom nearest
       neighbour interpolation or bilinear interpolation (of order 1)'

       This is an extract from the basemap module (truncated)
    """

    # Mesh Coordinates so that they are both 2D arrays
    xout,yout = np.meshgrid(xout,yout)

   # compute grid coordinates of output grid.
    delx = xin[1:]-xin[0:-1]
    dely = yin[1:]-yin[0:-1]

    xcoords = (len(xin)-1)*(xout-xin[0])/(xin[-1]-xin[0])
    ycoords = (len(yin)-1)*(yout-yin[0])/(yin[-1]-yin[0])


    xcoords = np.clip(xcoords,0,len(xin)-1)
    ycoords = np.clip(ycoords,0,len(yin)-1)

    # Interpolate to output grid using nearest neighbour
    if interpolation == 'NearestNeighbour':
        xcoordsi = np.around(xcoords).astype(np.int32)
        ycoordsi = np.around(ycoords).astype(np.int32)
        dataout = datain[ycoordsi,xcoordsi]

    # Interpolate to output grid using bilinear interpolation.
    elif interpolation == 'Bilinear':
        xi = xcoords.astype(np.int32)
        yi = ycoords.astype(np.int32)
        xip1 = xi+1
        yip1 = yi+1
        xip1 = np.clip(xip1,0,len(xin)-1)
        yip1 = np.clip(yip1,0,len(yin)-1)
        delx = xcoords-xi.astype(np.float32)
        dely = ycoords-yi.astype(np.float32)
        dataout = (1.-delx)*(1.-dely)*datain[yi,xi] + \
                  delx*dely*datain[yip1,xip1] + \
                  (1.-delx)*dely*datain[yip1,xi] + \
                  delx*(1.-dely)*datain[yi,xip1]

    return dataout