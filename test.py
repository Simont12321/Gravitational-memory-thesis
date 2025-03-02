import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fft2

# Define the grid in polar coordinates
r = np.linspace(0, 2 * np.pi, 200)
phi = np.linspace(0, 2 * np.pi, 200)
r, phi = np.meshgrid(r, phi)

# Compute the components of the vector field
F_r = 0.25 * np.sin(2 * r) * np.cos(2 * phi)
F_phi = -0.5 * np.sin(r) * np.sin(2 * phi) - 0.5 * np.sin(2 * r) * np.sin(2 * phi)

# Convert the polar coordinates to Cartesian for the grid
X = r * np.cos(phi)
Y = r * np.sin(phi)

# Compute the 2D Fourier transforms
F_r_fft = fftshift(fft2(F_r))
F_phi_fft = fftshift(fft2(F_phi))

# Get the magnitude of the Fourier transform for visualization
F_r_magnitude = np.abs(F_r_fft)
F_phi_magnitude = np.abs(F_phi_fft)

# Plot the magnitude of the Fourier transforms
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Fourier Transform of $F_r$")
plt.imshow(F_r_magnitude, extent=(-np.pi, np.pi, -np.pi, np.pi), cmap='inferno')
plt.colorbar(label='Magnitude')
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")

plt.subplot(1, 2, 2)
plt.title("Fourier Transform of $F_\phi$")
plt.imshow(F_phi_magnitude, extent=(-np.pi, np.pi, -np.pi, np.pi), cmap='inferno')
plt.colorbar(label='Magnitude')
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")

plt.tight_layout()
plt.show()
