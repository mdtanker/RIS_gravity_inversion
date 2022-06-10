# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:33:04 2019
function to calculate the derivative of gravity field of a vertical 
column using with respect to the depth if the top of the column
@author: fabiot
"""
# from the Hammer prism approximation, as shown on pg. 31, eq:2.19, of Jack McCubbine's thesis
from scipy import *
from numpy import *
def grav_column_der(x0,y0,z0,xc,yc,z1,z2,res,rho):
    r=sqrt((x0-xc)**2+(y0-yc)**2)
    # r1=r - sqrt(res/2)
    # r2=r + sqrt(res/2) 
    r1=r-0.5*res # fabio's original
    r2=r+0.5*res # fabio's original   
#    if r1 < 0:
#        r1=0
#        r2=0.5*res
#        f=4/pi
    r1[r1<0]=0 # will fail if prism is under obs point
    r2[r1<0]=0.5*res
    f=res**2/(pi*(r2**2-r1**2)) #eq 2.19 in McCubbine 2016 Thesis
    # anomaly_grad=0.0419*f*rho*(z1-z0)*(1/sqrt(r1**2+(z1-z0)**2)-1/sqrt(r2**2+(z1-z0)**2))  # switched r1 and r2  
    anomaly_grad=0.0419*f*rho*(z1-z0)*(1/sqrt(r2**2+(z1-z0)**2)-1/sqrt(r1**2+(z1-z0)**2)) # fabio's original

#    anomaly_grad=0.0419*f*rho*(z2-z0)*(1/sqrt(r1**2+(z2-z0)**2)-1/sqrt(r2**2+(z2-z0)**2))   
    return anomaly_grad    

"""
r1 and r2 = r +/- res/2, but should be +/- res/sqrt(2)
should be r1 - r2
what do these mean: r1[r1<0]=0 and r2[r1<0]=0.5*res

g = 0.0419 * f * rho * (r2 - r1 + sqrt(r1**2 + h**2) - sqrt(r2**2-h**2))
where f is a ratio factor for annulus
r2 is the outer radius
r1 is the inner radius 
and h is the height

dg/dh = 0.0419 * f * rho * h * [ (1/(sqrt(r1**2 + h**2))) - (1/(sqrt(r2**2 + h**2))) ]

with h = z1 - z0
dg/dz = 0.0419 * f * rho * (z1-z0) * [ (1/(sqrt(r1**2 + (z1-z0)**2))) - (1/(sqrt(r2**2 + (z1-z0)**2))) ]



"""

