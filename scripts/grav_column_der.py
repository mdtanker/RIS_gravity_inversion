import numpy as np

"""
from the Hammer prism approximation with annulus, as shown on pg. 31, eq:2.19, of Jack
McCubbine's thesis
"""


def grav_column_der(x0, y0, z0, xc, yc, z1, z2, res, rho):
    r = np.sqrt((x0 - xc) ** 2 + (y0 - yc) ** 2)
    # r1=r - sqrt(res/2)
    # r2=r + sqrt(res/2)
    r1 = r - 0.5 * res  # fabio's original
    r2 = r + 0.5 * res  # fabio's original
    r1[r1 < 0] = 0  # will fail if prism is under obs point
    r2[r1 < 0] = 0.5 * res
    f = res**2 / (np.pi * (r2**2 - r1**2))  # eq 2.19 in McCubbine 2016 Thesis
    # anomaly_grad=0.0419*f*rho*(z1-z0)*(1/sqrt(r1**2+(z1-z0)**2)-1/sqrt(r2**2+(z1-z0)\
    # **2))  # switched r1 and r2
    anomaly_grad = (
        0.0419
        * f
        * rho
        * (z1 - z0)
        * (
            1 / np.sqrt(r2**2 + (z1 - z0) ** 2)
            - 1 / np.sqrt(r1**2 + (z1 - z0) ** 2)
        )
    )  # fabio's original

    #    anomaly_grad=0.0419*f*rho*(z2-z0)*(1/sqrt(r1**2+(z2-z0)**2)-1/sqrt(r2**2+\
    # (z2-z0)**2))
    return anomaly_grad


def hammer_annulus_gravity(r, R, h, rho):
    """
    Gravity effect for an annulus of topography.
    Eq.2.13 in McCubbine 2016.

    r: inner radius of annulus in m
    R: outer radius of annulus in m
    h: height of annulus in m
    rho: density in g/cm^3 or Mg/m^3
    """

    anom = 0.0419 * rho * (R - r + np.sqrt(r**2 + h**2) - np.sqrt(R**2 + h**2))

    return anom

def hammer_annulus_gravity_elevated(r, R, h, e, rho):
    """
    Gravity effect for an annulus of topography at measurement height e above the geoid.
    Eq.6.12 in McCubbine 2016.

    r: inner radius of annulus in m
    R: outer radius of annulus in m
    h: height of annulus in m
    e: elevation of measurement point above the geoid in m
    rho: density in g/cm^3 or Mg/m^3
    """

    anom = 0.0419 * rho * (
        np.sqrt(R**2 + (e-h)**2) - np.sqrt(r**2 + (e-h)**2) - np.sqrt(R**2 + e**2))

    return anom

def hammer_prism_gravity(xp, yp, zp, res, rho):
    """
    Gravity effect for a prisms appoximated with an annulus of topography.
    Eq.2.20 in McCubbine 2016.
    Observation point at the origin.

    xp, yp, zp: coordinates of prism center in m
    res: resolution of the DEM in m
    rho: density in g/cm^3 or Mg/m^3
    """

    r = np.sqrt(xp**2 + yp**2) - np.sqrt(res**2/2) # eq 2.17
    R = np.sqrt(xp**2 + yp**2) + np.sqrt(res**2/2) # eq 2.18

    # f is ratio of area of terrain and area of annulus
    f = res**2 / (np.pi * (R**2 - r**2))  # eq 2.19 and 6.13

    anom = f * hammer_annulus_gravity(r=r, R=R, h=zp, rho=rho)

    return anom

def hammer_prism_gravity_elevated(xp, yp, zp, e, res, rho):
    """
    Gravity effect for a prisms appoximated with an annulus of topography.
    Eq.2.20 in McCubbine 2016.
    Observation point at the origin.

    xp, yp, zp: coordinates of prism center in m
    res: resolution of the DEM in m
    rho: density in g/cm^3 or Mg/m^3
    """

    r = np.sqrt(xp**2 + yp**2) - np.sqrt(res**2/2) # eq 2.17
    R = np.sqrt(xp**2 + yp**2) + np.sqrt(res**2/2) # eq 2.18

    # f is ratio of area of terrain and area of annulus
    f = res**2 / (np.pi * (R**2 - r**2))  # eq 2.19 and 6.13

    anom = f * hammer_annulus_gravity_elevated(r=r, R=R, h=zp, e=e, rho=rho)

    return anom








def hammer_annulus_gravity(rho, R, r, h):
    """ Gravity effect for an annulus of topography. Eq.2.13 in McCubbine 2016.
    rho: density in g/cm^3 or Mg/m^3
    R: outer radius of annulus in m
    r: inner radius of annulus in m
    h: height of annulus in m
    """
    anom = 0.0419 * rho * (R - r + np.sqrt(r**2 + h**2) - np.sqrt(R**2 + h**2))
    r = np.sqrt((x0 - xc) ** 2 + (y0 - yc) ** 2)
    r1= r - sqrt(res**2/2) # eq 2.17
    r2= r + sqrt(res**2/2) # eq 2.18

    f = res**2 / (np.pi * (r2**2 - r1**2))  # eq 2.19

    h = z1 - z0

    anom = f*0.0419*rho* (r2 - r1 + np.sqrt(r1**2 + (z1 - z0)**2) - np.sqrt(r2**2 + (z1 - z0)**2)) # eq 2.20

    #derivative of anomaly with respect to height
    anom_grad = f*0.0419*rho* (z1 - z0) * (1 / np.sqrt(r2**2 + (z1 - z0) ** 2)- 1 / np.sqrt(r1**2 + (z1 - z0) ** 2))

    return anom_grad
"""
Eq. 2.13 Gravity effect for an annulus with inner radius r, outer radius R, height h
g = 2 * pi * G * rho * (R - r + sqrt(r**2 + h**2) - sqrt(R**2 + h**2))
g = 0.0419 * rho * (R - r + sqrt(r**2 + h**2) - sqrt(R**2 + h**2))
with rho in units of g/cm^3

Gravity effect for an annulus with observation point at the origin
r = sqrt(x**2 + y**2) - sqrt(res**2/2)  (eq. 2.17)
R = sqrt(x**2 + y**2) + sqrt(res**2/2)  (eq. 2.18)
f = res**2 / (pi * (R**2 - r**2))  (eq. 2.19)
g = f * eq2.13


r1 and r2 = r +/- res/2, but should be +/- res/sqrt(2)
should be r1 - r2
what do these mean: r1[r1<0]=0 and r2[r1<0]=0.5*res

g = f *
g = 2*pi*G * f * rho * (r2 - r1 + sqrt(r1**2 + h**2) - sqrt(r2**2-h**2))
g = 0.0419 * f * rho * (r2 - r1 + sqrt(r1**2 + h**2) - sqrt(r2**2-h**2))
where f is a ratio factor for annulus
r2 is the outer radius
r1 is the inner radius
and h is the height

dg/dh = 0.0419 * f * rho * h * [ (1/(sqrt(r1**2 + h**2))) - (1/(sqrt(r2**2 + h**2))) ]

with h = z1 - z0
dg/dz = 0.0419 * f * rho * (z1-z0) * [ (1/(sqrt(r1**2 + (z1-z0)**2))) - (1/(sqrt(r2**2 \
    + (z1-z0)**2))) ]

"""
