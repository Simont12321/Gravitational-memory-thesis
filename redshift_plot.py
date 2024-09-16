import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x, y, del_x, del_y):

    return -0.25 * (y-del_y)**2 + 0.5 * (x-del_x)**2 * (y-del_y)**2 # first approximation 

    # full coordinate translation expresion 
    # return -0.25 * (y**2 + 2*y*(delta) + (delta)**2) + 0.5*(x**2 + 2*x*(delta) + (delta)**2)*(y**2 + 2*y*(delta) + (delta)**2)

    ## Higher- order terms, up to x**4 * y**4
    #return 0.5*(-(y**2)/2 + (x**2 * y**2) - (x**4 * y**2)/3 + (y**4)/24 - (x**2 * y**4)/12 + (x**4 * y**4)/36)

    #return 0.5*(-(y**2)/2 + (x**2 * y**2) + (y**4)/24) # max is y**4 

# GW displacement 
del_x = 1
del_y = 0.5

# Create a grid of x and y values from -2 to 2
max_value = 3
x = np.linspace(-max_value, max_value, 400)
y = np.linspace(-max_value, max_value, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y, del_x, del_y)

# Create the filled contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='RdBu')
zero_contour = plt.contour(X, Y, Z, levels=[0], colors='lime', linewidths=2)

# Add labels and a color bar
plt.title(f'Plot of initial redshift approximation for a GW at ({del_x}, {del_y})')  # $z = -\frac{1}{4}y^2 + \frac{1}{2}x^2y^2$
plt.xlabel(r'$\theta_x$')
plt.ylabel(r'$\theta_y$')
plt.colorbar(contour, label='Scale in terms of $h_M$')


# Add a label for the zero contour
plt.clabel(zero_contour, fmt='%1.1f', colors='black')

plt.grid(True)
plt.show()
