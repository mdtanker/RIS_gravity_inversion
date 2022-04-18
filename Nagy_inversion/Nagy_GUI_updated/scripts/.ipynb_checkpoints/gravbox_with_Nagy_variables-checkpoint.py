# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:31:58 2013

function to calculate the gravity field of a prism
@author: fabiot
"""
import scipy as sp
import numpy as np

def gravbox_with_Nagy_variables( XG, YG, ZG, XLayer, YLayer, ZLayer, ZLower_Layer, RESLayer, RHO):
    a  = abs(XG-XLayer)

    b  = abs(YG-YLayer)

    a1 = a - (RESLayer/2)
    a2 = a + (RESLayer/2)
    b1 = b - (RESLayer/2)
    b2 = b + (RESLayer/2)
    
    c1 = abs(ZG - ZLayer)

    c2 = abs(ZG - ZLower_Layer)

    ###############################################

    r111=sp.sqrt(a1**2 + b1**2 + c1**2)
    r112=sp.sqrt(a1**2 + b1**2 + c2**2)
    r121=sp.sqrt(a1**2 + b2**2 + c1**2)
    r122=sp.sqrt(a1**2 + b2**2 + c2**2)
    r211=sp.sqrt(a2**2 + b1**2 + c1**2)
    r212=sp.sqrt(a2**2 + b1**2 + c2**2)
    r221=sp.sqrt(a2**2 + b2**2 + c1**2)
    r222=sp.sqrt(a2**2 + b2**2 + c2**2)

    eq1 =    (c1**2 + b2**2 + b2*r221) / (    (b2 + r221) * sp.sqrt(b2**2 + c1**2)   )
    ###############################################
    eq2 =    (c1**2 + b2**2 + b2*r121) / (    (b2 + r121) * sp.sqrt(b2**2 + c1**2)   )
    ###############################################
    eq3 =    (c1**2 + b1**2 + b1*r211) / (    (b1 + r211) * sp.sqrt(b1**2 + c1**2)   )
    ###############################################
    eq4 =    (c1**2 + b1**2 + b1*r111) / (    (b1 + r111) * sp.sqrt(b1**2 + c1**2)   )
    ###############################################    
    eq5 =    (c2**2 + b2**2 + b2*r222) / (    (b2 + r222) * sp.sqrt(b2**2 + c2**2)   )
    ###############################################    
    eq6 =    (c2**2 + b2**2 + b2*r122) / (    (b2 + r122) * sp.sqrt(b2**2 + c2**2)   )
    ###############################################
    eq7 =    (c2**2 + b1**2 + b1*r212) / (    (b1 + r212) * sp.sqrt(b1**2 + c2**2)   )
    ###############################################    
    eq8 =    (c2**2 + b1**2 + b1*r112) / (    (b1 + r112) * sp.sqrt(b1**2 + c2**2)   )
    ###############################################  
        
    anomaly = 0.00667 * RHO * (
    a2 * (sp.log((b2+r221)/(b2+r222)) + sp.log((b1+r212)/(b1+r222))) +
    a1 * (sp.log((b2+r122)/(b2+r121)) + sp.log((b1+r111)/(b1+r121))) +
    
    b2 * (sp.log((a2+r221)/(a2+r222)) + sp.log((a1+r122)/(a1+r121))) +
    b1 * (sp.log((a2+r212)/(a2+r211)) + sp.log((a1+r111)/(a1+r112))) +
    
    c1 * sp.arcsin (eq1) -    
    c1 * sp.arcsin (eq2) - 

    c1 * sp.arcsin (eq3) +
    c1 * sp.arcsin (eq4) -

    c2 * sp.arcsin (eq5) +
    c2 * sp.arcsin (eq6) +

    c2 * sp.arcsin (eq7) -
    c2 * sp.arcsin (eq8))

    return anomaly

    
