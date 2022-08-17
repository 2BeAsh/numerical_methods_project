#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#%% Forces
def prey_force(t, x_vec, a):
    x, y = x_vec
    N = x.size
    fx, fy = np.empty(N), np.empty(N) # Empty derivative arrays to be updated

    # For each element in x and y, calculate its derivative
    for j in range(N): # IF WE CAN DO THIS WITHOUT A LOOP IT WOULD BE GREAT
        # Exclude k = j
        idx = np.where(x != x[j]) # Same index for both x and y
        x_k = x[idx]
        y_k = y[idx]
        
        x_expr = (x[j] - x_k) / (np.abs(x[j] - x_k))**2 - a * (x[j] - x_k) # NEED PREDATOR
        y_expr = (y[j] - y_k) / (np.abs(y[j] - y_k))**2 - a * (y[j] - y_k)

        fx[j] = np.sum(x_expr) / N
        fy[j] = np.sum(y_expr) / N

    return fx, fy


#%% Create Swarm
def prey_movement(N, L, t_end, dt):
    """
    N: int - Amount of prey
    L: float - Starting area which prey can be in
    t_end: float - How long the simulation should run
    dt: time step between each update
    """
    
    t_vals = np.arange(0, t_end, dt)
    x = np.empty(N, t_vals.size)
    y = np.empty(N, t_vals.size)
        
    # Create uniformly distributed initial positions
    x[0]= np.random.uniform(low=-L, high=L, size=N)
    y[0]= np.random.uniform(low=-L, high=L, size=N)


    # Use Euler method to update
    for t in t_vals:
        F = prey_force(t=t, x_vec=(x, y), a=1)        
        x[t] = x[t] + F * dt
        y[t] = y[t] + F * dt
        
    return x, y

