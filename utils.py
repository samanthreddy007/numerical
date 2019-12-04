import numpy as np
import matplotlib.pyplot as plt
def DLU(A) :
    """ 
    Given a matrix, decomposes it into a diagonal matrix, upper matrix and lower matrix
    """
    m = np.shape(A)
    D = np.zeros(m)
    L = np.zeros(m)
    U = np.zeros(m)
    
    for i in range(m[0]):
        D[i,i] = A[i,i]
        for j in range(0,i):
            L[i,j] = A[i,j]
        for j in range(i+1,m[0]):
            U[i,j] = A[i,j]

    return D,-L,-U


def plot_x(all_x) :
    """
    Plot the given arrays
    """

    plt.figure(figsize=(16,9))
    for i in range(0,len(all_x)-1):
        line = plt.plot( [all_x[i][0],all_x[i+1][0]], [all_x[i][1],all_x[i+1][1]] )
        # print(f"Plotting {[all_x[i][0],all_x[i+1][0]]} {[all_x[i][1],all_x[i+1][1]]}")
        add_arrow(line)
    plt.show()

def add_arrow(line,i = 0, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line[i].get_color()

    xdata = line[i].get_xdata()
    ydata = line[i].get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = (start_ind + 1) % 2
    else:
        end_ind = (start_ind - 1 ) % 2
    
    line[i].axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )