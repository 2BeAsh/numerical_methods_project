#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#%% Forces
def force(t, coord, a, b, c, p):
    """
    Parameters
    ----------
    t : float
        Time
    coord : Tuple of two 1darrays each with two 1darrays
        first element is prey coordinates [x, y] and second is predator coordinates [zx, zy]
    a : float
        Prey Attraction paramater.
    b : float
        Predator's repulsion on prey paramater.
    c : float
        Efficiency of predator parameter
    p : float
        Predator parameter
        
        

    Returns
    -------
    fx : 1darray
        Each prey particle´s derivative in the x-direction
    fy : 1darray
        Each prey particle´s derivative in the y-direction
    fzx: 1darray
        The predator´s derivative in the x-direction
    fzy: 1darray
        The predator´s derivative in the y-direction

    """

    prey_coord, predator_coord = coord
    x, y = prey_coord
    zx, zy = predator_coord
    
    N = x.size
    fx, fy = np.empty(N), np.empty(N) # Empty derivative arrays to be updated
    
    # For each element in x and y, calculate its derivative
    for j in range(N): # IF WE CAN DO THIS WITHOUT A LOOP IT WOULD BE GREAT
        # Exclude k = j
        idx = np.where(x != x[j]) # Same index for both x and y
        x_k = x[idx]
        y_k = y[idx]
        
        dist_prey = np.abs(x[j] - x_k 
                         + y[j] - y_k) # Distance between prey and prey
        dist_predator = np.abs(x[j] - zx
                              +y[j] - zy) # Distance between prey and predator

        x_expr  = (x[j] - x_k) / dist_prey**2 - a * (x[j] - x_k) + b * (x[j] - zx) / dist_predator**2 # Expression inside sum in Equation 1.1 in Y. Chen
        y_expr  = (y[j] - y_k) / dist_prey**2 - a * (y[j] - y_k) + b * (y[j] - zy) / dist_predator**2

        # Append
        fx[j]  = 1 / N * np.sum(x_expr)
        fy[j]  = 1 / N * np.sum(y_expr)
        
    # Predator - No need for loop since only 1 particle IF WE NEED MULTIPLE PREDATORS, WE NEED A SEPERATE LOOP FOR THIS
    fzx = c / N * (x - zx) / np.abs(x - zx + y - zy)**p
    fzy = c / N * (y - zy) / np.abs(x - zx + y - zy)**p
    

    return fx, fy, fzx, fzy

#%% Test if force function works on simple example
prey_coord = [np.arange(10), np.arange(10)]
pred_coord = np.array([0.5,0.1])
print(force(t=1, coord=(prey_coord, pred_coord), a=1, b=1, c=2, p=3))

#%% Create movement of prey and predator
def movement(N, L, t_end, dt, a, b, c, p):
    """
    N: int - Amount of prey
    L: float - Starting area which prey can be in
    t_end: float - How long the simulation should run
    dt: time step between each update
    """
    # Initial values for prey and then predator
    # Prey are uniformly distributed in a square with side L
    t_vals = np.arange(0, t_end, dt)
    x = [np.random.uniform(size=N)]
    y = [np.random.uniform(size=N)]
    
    # Predator starts in the middle of square
    zx = [0]
    zy = [0]
    
    for t in t_vals:
        prey_coord = [x[-1], y[-1]]
        predator_coord = [zx[-1], zy[-1]]
        fx, fy, fzx, fzy = force(t, coord=(prey_coord, predator_coord), a, b, c, p)
    
    return x, y

#%% Test movement function

print(movement(N=10, L=4, t_end=20, dt=1, a=1, b=1, c=1, p=1))






