import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x, y):
    delta = 0.5  # coordinate translation 

    return -0.25 * (y+0)**2 + 0.5 * (x+0)**2 * (y+0)**2 # first approximation 

    # full coordinate translation expresion 
    # return -0.25 * (y**2 + 2*y*(delta) + (delta)**2) + 0.5*(x**2 + 2*x*(delta) + (delta)**2)*(y**2 + 2*y*(delta) + (delta)**2)

    ## Higher- order terms, up to x**4 * y**4
    #return 0.5*(-(y**2)/2 + (x**2 * y**2) - (x**4 * y**2)/3 + (y**4)/24 - (x**2 * y**4)/12 + (x**4 * y**4)/36)

    #return 0.5*(-(y**2)/2 + (x**2 * y**2) + (y**4)/24) # max is y**4 


# Create a grid of x and y values from -2 to 2
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create the filled contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='RdBu')
zero_contour = plt.contour(X, Y, Z, levels=[0], colors='lime', linewidths=2)

# Add labels and a color bar
plt.title(r'Plot of initial redshift $z = -\frac{1}{4}y^2 + \frac{1}{2}x^2y^2$')
plt.xlabel(r'$\theta_x$')
plt.ylabel(r'$\theta_y$')
plt.colorbar(contour, label='Scale in terms of $h_M$')

# Add a label for the zero contour
plt.clabel(zero_contour, fmt='%1.1f', colors='black')

plt.grid(True)
plt.show()
