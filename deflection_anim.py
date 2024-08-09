import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

# Define the grid for x and y
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Define the vector field function
def vector_field(B, X, Y):
    U = X * Y - B * X * Y
    V = -0.5 * Y + 0.5 * B * Y #+ (X**2 * Y - B * X**2 * Y) #higher order terms in ()
    return U, V

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel(r'$\theta_x$')
ax.set_ylabel(r'$\theta_y$')
ax.set_title('Flat sky angular deflection approximation as a function of '+ r'$\beta$')

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
