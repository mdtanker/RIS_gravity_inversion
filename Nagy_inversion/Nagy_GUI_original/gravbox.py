# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:31:58 2013

function to calculate the gravity field of a prism
@author: fabiot
"""
#from scipy import *
#from numpy import *
import scipy as sp

def gravbox(x0,y0,z0,x1,x2,y1,y2,z1,z2,rho):
    u1=(x0-x1)
    u2=(x0-x2);
    v1=(y0-y1);
    v2=(y0-y2);
    w1=(z0-z1);
    w2=(z0-z2);
    r111=sp.sqrt(u1**2+v1**2+w1**2);
    r112=sp.sqrt(u1**2+v1**2+w2**2);
    r121=sp.sqrt(u1**2+v2**2+w1**2);
    r122=sp.sqrt(u1**2+v2**2+w2**2);
    r211=sp.sqrt(u2**2+v1**2+w1**2);
    r212=sp.sqrt(u2**2+v1**2+w2**2);
    r221=sp.sqrt(u2**2+v2**2+w1**2);
    r222=sp.sqrt(u2**2+v2**2+w2**2);
    anomaly=0.00667*rho*(u1*(sp.log((v2+r122)/(v2+r121))-sp.log((v1+r112)/(v1+r111)))-u2*(sp.log((v2+r222)/(v2+r221))-sp.log((v1+r212)/(v1+r211)))+   
    v1*(sp.log((u2+r212)/(u2+r211))-sp.log((u1+r112)/(u1+r111)))-v2*(sp.log((u2+r222)/(u2+r221))-sp.log((u1+r122)/(u1+r121)))+
    w1*(sp.arctan((u1*v2)/(w1*r121))+sp.arctan((u2*v1)/(w1*r211))-sp.arctan((u1*v1)/(w1*r111))-sp.arctan((u2*v2)/(w1*r221)))+
    w2*(sp.arctan((u1*v1)/(w2*r112))+sp.arctan((u2*v2)/(w2*r222))-sp.arctan((u1*v2)/(w2*r122))-sp.arctan((u2*v1)/(w2*r212))))         
    return anomaly