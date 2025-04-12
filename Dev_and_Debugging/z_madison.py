import numpy as np
import matplotlib.pyplot as plt

# Define ranges for theta and phi
theta_values = np.linspace(-2, 2, 200)  # Range of theta from -2 to 2
phi_values = np.linspace(-2, 2, 200)    # Range of phi from -2 to 2

# Create a meshgrid for theta and phi
Theta, Phi = np.meshgrid(theta_values, phi_values)

# Calculate the expression
expression = 0.25 * (np.exp(-2j * (Phi + 0)) + np.exp(2j * (Phi + 0))) * (1 - np.cos((Theta + 0)))

# Separate the real and imaginary parts for plotting
real_part = expression.real
imaginary_part = expression.imag

# Plotting the real part
plt.figure(figsize=(12, 6))

contour_real = plt.contourf(Phi, Theta, real_part, levels=50, cmap='viridis')
colorbar = plt.colorbar(contour_real)
colorbar.set_label('Scale in terms of $h_M$', fontsize=12)

contour_zero_real = plt.contour(Phi, Theta, real_part, levels=[0], colors='red', linewidths=2)
plt.clabel(contour_zero_real, fmt='%1.1f', colors='red')
plt.title('Initial redshift [Madison 2021 eq.1]')
plt.xlabel('φ')
plt.ylabel('θ')


plt.tight_layout()
plt.show()
