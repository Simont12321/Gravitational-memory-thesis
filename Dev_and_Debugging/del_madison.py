import numpy as np
import matplotlib.pyplot as plt

# Define the vector field in spherical coordinates
def vector_field(theta, phi):
    # Calculate the components of the vector field
    term1 = -0.25 * np.sin(theta) * (np.exp(2j * phi) + np.exp(-2j * phi))
    term2 = 1j * (np.exp(2j * phi) - np.exp(-2j * phi))
    term3 = 0.25 * np.sin(theta) * ((1 + np.cos(theta)) * (np.exp(2j * phi) + np.exp(-2j * phi)))
    term4 = 2j * (np.exp(2j * phi) - np.exp(-2j * phi))

    # Calculate the vector components
    V_theta = term1 + term3
    V_phi = term2 + term4

    return V_theta, V_phi

# Convert spherical to Cartesian
def spherical_to_cartesian(V_theta, V_phi, theta, phi):
    # Components in Cartesian coordinates
    Vx = V_theta * np.cos(theta) * np.cos(phi) - V_phi * np.sin(phi)
    Vy = V_theta * np.cos(theta) * np.sin(phi) + V_phi * np.cos(phi)
    return Vx, Vy

# Create a grid in the (x, y) plane
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Convert to spherical coordinates (assuming z = 0, theta = pi/2)
R = np.sqrt(X**2 + Y**2)
Phi = np.arctan2(Y, X)
Theta = np.pi / 2

# Calculate the vector field in Cartesian coordinates
V_theta, V_phi = vector_field(Theta, Phi)
U, V = spherical_to_cartesian(V_theta, V_phi, Theta, Phi)

# Plot the vector field with scaled-down vectors
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U.real, V.real, color='b', angles='xy', scale_units='xy', scale=10)  # Increase the scale factor
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title("Vector Field Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()
