# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:33:04 2019
function to calculate the derivative of gravity field of a vertical column using with respect to the depth if the top of the column
@author: fabiot
"""

from scipy import *
from numpy import *
def grav_column_der(x0,y0,z0,xc,yc,z1,z2,res,rho):
    r=sqrt((x0-xc)**2+(y0-yc)**2)
    r1=r-0.5*res
    r2=r+0.5*res    
#    if r1 < 0:
#        r1=0
#        r2=0.5*res
#        f=4/pi
    r1[r1<0]=0
    r2[r1<0]=0.5*res
    f=res**2/(pi*(r2**2-r1**2))
    anomaly_grad=0.0419*f*rho*(z1-z0)*(1/sqrt(r2**2+(z1-z0)**2)-1/sqrt(r1**2+(z1-z0)**2))   
#    anomaly_grad=0.0419*f*rho*(z2-z0)*(1/sqrt(r1**2+(z2-z0)**2)-1/sqrt(r2**2+(z2-z0)**2))   
    return anomaly_grad    


