#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#%% Forces
def prey_prey_deriv(t, x_vec, a):
    x, y = x_vec
    N = x.size
    fx, fy = np.empty(N), np.empty(N) # Empty derivative arrays to be updated

    # For each element in x and y, calculate its derivative
    for j in range(N):
        x_expr = (x[j] - x) / (np.abs(x[j] - x))**2 - a * (x[j] - x)
        y_expr = (y[j] - y) / (np.abs(y[j] - y))**2 - a * (y[j] - y)

        fx[j] = np.sum(x_expr) / N
        fy[j] = np.sum(y_expr) / N

    return fx, fy




#%% Create Swarm
def swarm(N, L, t_end):
    """
    N: int - Amount of prey
    L: float - Starting area which prey can be in
    """
    #Create uniformly distributed random positions
    x_vec = np.random.uniform(low=-L, high=L, size=(N, 2))
    print(x_vec.shape)
    t = 0

    # Use Euler method to update
    while t <= t_end:
        x = x_vec[0]
        y = x_vec[1]
        fx, fy = prey_prey_deriv(t=t, x_vec=(x,y), a=1)
        print(fx)


    return x


print(swarm(N=5, L=1, t_end=20))
