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

        dist_prey_p2 = (x[j] - x_k)**2 + (y[j] - y_k)**2 # Distance between prey and prey squared
        dist_prey_p2 = np.where(dist_prey_p2 == 0, 0.05, dist_prey_p2) # Add minimum distance between particles

        dist_predator = ((x[j] - zx)**2 + (y[j] - zy)**2)**(1/2) # Distance between prey and predator
        # dist_predator = np.where(dist_predator == 0, 0.05, dist_predator)

        x_expr  = (x[j] - x_k) / dist_prey_p2 - a * (x[j] - x_k)  # Expression inside sum in Equation 1.1 in Y. Chen
        y_expr  = (y[j] - y_k) / dist_prey_p2 - a * (y[j] - y_k) 

        # Append
        fx[j]  = 1 / N * np.sum(x_expr) + b * (x[j] - zx) / dist_predator**2
        fy[j]  = 1 / N * np.sum(y_expr) + b * (y[j] - zy) / dist_predator**2

    # Predator - No need for loop since only 1 particle IF WE NEED MULTIPLE PREDATORS, WE NEED A SEPERATE LOOP FOR THIS
    # Since predator and prey may colide, add minimal distance
    dist_loc = ((x - zx)**2 + (y - zy)**2)**(1/2)
    # dist_loc = np.where(dist_loc == 0, 0.05, dist_loc)
    fzx = c / N * np.sum((x - zx) / dist_loc**p)
    fzy = c / N * np.sum((y - zy) / dist_loc**p)

    return fx, fy, fzx, fzy

#%% Getting rid of the for loop - Temporary
def force_no_loop(t, coord, a, b, c, p):
    prey_coord, pred_coord = coord
    x, y = prey_coord
    zx, zy = pred_coord
    N = x.size
    
    # x and y broadcasting
    xj = x[:, None]
    xk = x[None, :]

    yj = y[:, None]
    yk =y[None, :]

    dist_prey_pred = (x - zx)**2 + (y - zy)**2
    dist_xy_p2     = (xj - xk)**2 + (yj - yk)**2 
    
    # Get x and y specific, set terms with distance 0 equal to 0
    div_term_x = np.divide(xj - xk, dist_xy_p2, out=np.zeros((N,N), dtype=float), where=dist_xy_p2!=0)
    div_term_y = np.divide(yj - yk, dist_xy_p2, out=np.zeros((N,N), dtype=float), where=dist_xy_p2!=0)

    #div_term_x = np.where(dist_xy_p2 == 0, 0, (xj - xk) / dist_xy_p2)
    #div_term_y = np.where(dist_xy_p2 == 0, 0, (yj - yk) / dist_xy_p2)
    
    fx = 1 / N * np.sum(div_term_x - a * (xj - xk), axis=1) \
                  + b * (x - zx) / dist_prey_pred
    
    fy = 1 / N * np.sum(div_term_y - a * (yj - yk), axis=1) \
                  + b * (y - zy) / dist_prey_pred
    
    fzx = c / N * np.sum((x - zx) / dist_prey_pred**(p/2)) # Divide p by 2 because normally squared
    fzy = c / N * np.sum((y - zy) / dist_prey_pred**(p/2))
    return fx, fy, fzx, fzy


#%% Test if force function works on simple example
prey_coord = [np.arange(16), np.arange(16)]
pred_coord = np.array([0.5,0.1])
#print(force(t=1, coord=(prey_coord, pred_coord), a=1, b=1, c=2, p=3)[1])
#print(force_no_loop(t=1, coord=(prey_coord, pred_coord), a=1, b=1, c=2, p=3)[1])

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
    x = np.random.uniform(low=-L, high=L, size=N)
    y = np.random.uniform(low=-L, high=L, size=N)
    fx = np.random.random(size=N)
    fy = np.random.random(size=N)

    zx = 0
    zy = 0
    x_list = [x]
    y_list = [y]

    fx_list = [fx]
    fy_list = [fy]

    # Predator starts with zero velocity
    fzx_list = [0]
    fzy_list = [0]

    # Predator starts in the middle of square
    zx_list = [zx]
    zy_list = [zy]

    N_list = [N]

    for t in t_vals:
        # Get values
        prey_coord = [x, y]
        predator_coord = [zx, zy]
        fx, fy, fzx, fzy = force_no_loop(t=t, coord=(prey_coord, predator_coord), a=a, b=b, c=c, p=p)
        # Update values
        x = x + fx * dt
        y = y + fy * dt
        zx = zx + fzx * dt
        zy = zy + fzy * dt
        
       
        
        x = np.clip(x, -1, 1)
        y = np.clip(y, -1, 1)
        
        if x<= -0.9 or x>=0.9:
            fx = 0
            fzx = 0
        if y<= -0.9 or y>=0.9:
            fy = 0
            fzy = 0
        
        
        # Distance between pred and prey
        r_min = 0.1
        r = np.sqrt((x-zx)**2 + (y-zy)**2)
        
        
        x = x[r > r_min]
        y = y[r > r_min]
        fx = fx[r > r_min]
        fy = fy[r > r_min]

        # Append values
        x_list.append(x)
        y_list.append(y)
        zx_list.append(zx)
        zy_list.append(zy)
        fx_list.append(fx)
        fy_list.append(fy)
        fzx_list.append(fzx)
        fzy_list.append(fzy)
        N_list.append(len(x))

    return x_list, y_list, zx_list, zy_list, fx_list, fy_list, fzx_list, fzy_list, N_list


#%% Test movement function
x, y, zx, zy, fx, fy, fzx, fzy, N_living =  movement(N=200, L=1, t_end=5, dt=0.2, a=1, b=0.2, c=4, p=2.5)
# print(N_living)

#%%

# Normalize vector fx, fy...
for i in range(len(x)):
    f_len = np.sqrt(fx[i]**2 + fy[i]**2)
    fz_len = np.sqrt(fzx[i]**2 + fzy[i]**2)
    fx[i] = fx[i]/f_len
    fy[i] = fy[i]/f_len
    fzx[i] = fzx[i]/fz_len
    fzy[i] = fzy[i]/fz_len




for i in range(len(x)):
    plt.figure(dpi=150)
    plt.title(f"PP, t={i}, {N_living[i]}")
    plt.scatter(x[i], y[i], s=4)
    plt.scatter(zx[i], zy[i], color="r", s=4)
    plt.quiver(x[i], y[i], fx[i] , fy[i])
    plt.quiver(zx[i], zy[i], fzx[i] , fzy[i], color="r")
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
