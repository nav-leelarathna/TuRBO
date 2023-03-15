import sys
sys.path.append('../')
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import numpy as np
import matplotlib.pyplot as plt
from bo.functions import Levy 


def visualise_ackley():
    def objective(x, y):
        return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 * 
    pi * x)+cos(2 * pi * y))) + e + 20


    r_min, r_max = -32.768, 32.768
    xaxis = arange(r_min, r_max, 2.0)
    yaxis = arange(r_min, r_max, 2.0)
    x, y = meshgrid(xaxis, yaxis)
    results = objective(x, y)
    figure = plt.figure()
    # axis = figure.gca( projection='3d')
    axis = figure.add_subplot(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet', shade= "false")
    plt.show()
    plt.contour(x,y,results)
    plt.show()
    plt.scatter(x, y, results)
    plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def plot_ackley_3d():
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(projection='3d')

    # Make data.
    # X = np.arange(-32, 32, 0.25)
    X = np.arange(-5, 5, 0.1)
    Y = np.arange(-5, 5, 0.1)
    # Y = np.arange(-32, 32, 0.25)
    X, Y = np.meshgrid(X, Y)

    a = 20
    b = 0.2
    c = 2 * np.pi

    sum_sq_term = -a * np.exp(-b * np.sqrt(X*X + Y*Y) / 2)
    cos_term = -np.exp((np.cos(c*X) + np.cos(c*Y)) / 2)
    Z = a + np.exp(1) + sum_sq_term + cos_term

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z,cmap='jet',linewidth=1, antialiased=False)
    # surf = ax.plot_surface(X, Y, Z, linewidth=0, cmap='viridis', edgecolor='none')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig("plots/ackleyVis.png")
    plt.show()

def plotLevy():
    def f(a,b):
        x = np.array([a,b])
        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[0]) ** 2 + \
            np.sum((w[1:2- 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:2 - 1] + 1) ** 2)) + \
            (w[2 - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[2 - 1])**2)
        return val
    levy = Levy(2)

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(projection='3d')

    # Make data.
    # X = np.arange(-32, 32, 0.25)
    X = np.arange(-5, 10, 0.5)
    Y = np.arange(-5, 10, 0.5)
    N = X.size
    # Y = np.arange(-32, 32, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((N,N))

    for i in range(N):
        for j in range(N): 
                Z[i,j] = f(X[i,j],Y[i,j])

    print(Z.shape)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z,cmap='jet',linewidth=5, antialiased=False)
    # surf = ax.plot_surface(X, Y, Z, linewidth=0, cmap='viridis', edgecolor='none')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig("visualisation/plots/levyVis.png")
    plt.show()


if __name__ == "__main__":
    # plot_ackley_3d()
    plotLevy() 
    # visualise_ackley()