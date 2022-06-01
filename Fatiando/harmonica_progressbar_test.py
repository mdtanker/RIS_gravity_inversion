from click import progressbar
import matplotlib.pyplot as plt
import numpy as np
import verde as vd

import harmonica as hm

# Create a layer of prisms
region = (0, 100e3, -40e3, 40e3)
spacing = 2e3
(easting, northing) = vd.grid_coordinates(region=region, spacing=spacing)
surface = 100 * np.exp(-((easting - 50e3) ** 2 + northing ** 2) / 1e9)
density = 2670.0 * np.ones_like(surface)
prisms = hm.prism_layer(
    coordinates=(easting[0, :], northing[:, 0]),
    surface=surface,
    reference=0,
    properties={"density": density},
)

# Compute gravity field of prisms on a regular grid of observation points
coordinates = vd.grid_coordinates(region, spacing=spacing, extra_coords=1e3)
gravity = prisms.prism_layer.gravity(
    coordinates, field="g_z", 
    progressbar=True,
        )

# Plot gravity field
plt.pcolormesh(*coordinates[:2], gravity)
plt.colorbar(label="mGal", shrink=0.8)
plt.gca().set_aspect("equal")
plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
plt.title("Gravity acceleration of a layer of prisms")
plt.xlabel("easting [m]")
plt.ylabel("northing [m]")
plt.tight_layout()
plt.show()

potential = prisms.prism_layer.gravity(
    coordinates, field="potential", 
    # progressbar=True,
        )

# Plot gravity field
plt.pcolormesh(*coordinates[:2], potential)
plt.colorbar(label="mGal", shrink=0.8)
plt.gca().set_aspect("equal")
plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
plt.title("Gravity potential of a layer of prisms")
plt.xlabel("easting [m]")
plt.ylabel("northing [m]")
plt.tight_layout()
plt.show()

test = prisms.prism_layer.gravity(
    coordinates, field="potential", 
    progressbar=True,
        )

# with 500m spacing, 32361 iterations, took 3:39, 147.11it/s
# after adding in update_progressbar check, 3:25, 