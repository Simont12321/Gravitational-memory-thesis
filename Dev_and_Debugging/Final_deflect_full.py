import numpy as np
import matplotlib.pyplot as plt

# Define the grid in polar coordinates
r = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
r, phi = np.meshgrid(r, phi)

# Compute the components of the vector field
F_r = 0.25 * np.sin(2 * r) * np.cos(2 * phi)
F_phi = -0.5 * np.sin(r) * np.sin(2 * phi) - 0.5 * np.sin(2 * r) * np.sin(2 * phi)

# Plot the vector field in polar coordinates
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Quiver plot in polar coordinates
ax.quiver(phi, r, F_r, F_phi, angles='xy', scale=50, color='b')

ax.set_title('Final Deflection Vector Field Plot in Polar Coordinates')
plt.show()
