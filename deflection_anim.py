import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

# Define the grid for x and y
max_value = 3
x = np.linspace(-max_value, max_value, 20)
y = np.linspace(-max_value, max_value, 20)
X, Y = np.meshgrid(x, y)

# coordinate translation
delx = 0.7
dely = 0.3

# Define the vector field function
def vector_field(B, X, Y):
    U = (X+delx) * (Y+dely) - B * (X+delx) * (Y+dely)
    V = -0.5 * (Y+dely) + 0.5 * B * (Y+dely) #+ (X**2 * Y - B * X**2 * Y) #higher order terms in ()
    return U, V

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-max_value, max_value)
ax.set_ylim(-max_value, max_value)
ax.set_xlabel(r'$\theta_x$', fontsize=20)
ax.set_ylabel(r'$\theta_y$', fontsize=20)
ax.set_title('Flat sky angular deflection approximation as a function of '+ r'$\beta$', fontsize = 15)

# Set the size of the axes tick marks
ax.tick_params(axis='both', which='major', labelsize=15)

# Initial vector field
B0 = 0
U, V = vector_field(B0, X, Y)
quiver = ax.quiver(X, Y, U, V, color='b')

# Update function for animation
def update(B):
    U, V = vector_field(B, X, Y)
    quiver.set_UVC(U, V)
    return quiver,

# Add a slider for the B parameter
ax_B_slider = plt.axes([0.95, 0.25, 0.03, 0.5])  # Adjust position for vertical slider
B_slider = Slider(ax_B_slider, r'$\beta$', 0, 2, valinit=B0, orientation='vertical')

# Increase slider font size
B_slider.label.set_fontsize(15)
B_slider.valtext.set_fontsize(15)

# Set the update function for the slider
B_slider.on_changed(update)


"""
### Create an animation, comment out the slider lines above 
ani = FuncAnimation(
    fig, update, frames=np.linspace(0, 2, 100), interval=100, blit=True, repeat=True
)

### To save the animation as a gif
writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save('deflections.gif', writer=writer)
"""

plt.grid()
plt.show()
