import numpy as np
import matplotlib.pyplot as plt

# Define the polar function h(r, phi)
def h_func(r, phi):
    return (1/4) * (2 * np.cos(2 * phi) - 2 * np.cos(2 * phi) * np.cos(r))

# Set the maximum value of r
r_max = 10

# Create an array of r and phi values
r = np.linspace(0, r_max, 400)
phi = np.linspace(0, 2 * np.pi, 400)

# Create a meshgrid for r and phi
R, Phi = np.meshgrid(r, phi)

# Calculate h(r, phi) for all (r, phi) pairs
Z = h_func(R, Phi)

# Create a polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))

# Plot the data using pcolormesh
c = ax.pcolormesh(Phi, R, Z, shading='auto', cmap='inferno')

# Add a colorbar to show the scale
cbar = fig.colorbar(c, ax=ax, label='Scale in terms of $h_M$')
cbar.ax.tick_params(labelsize=14)  # Set colorbar tick label size
cbar.set_label('Scale in terms of $h_M$', fontsize=14)  # Set colorbar label size

# Set the title
ax.set_title('Heatmap of initial redshift in Polar Coordinates', fontsize=16)
ax.set_xlabel('ϕ (degrees)', fontsize = 14)

# Set radial ticks (you can adjust the number and position of labels as needed)
ax.set_rlabel_position(-22.5)  # Positioning for better appearance
label_position = ax.get_rlabel_position()
ax.text(np.radians(label_position + 15), ax.get_rmax() / 2., 'θ (degrees)',
        rotation=label_position, ha='center', va='center', fontsize=14)

# Set angular ticks (you can customize the labels and number of ticks)
ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))

# Set radial ticks and labels
ax.tick_params(axis='both', which='major', labelsize=12)

# Display the plot
plt.show()
