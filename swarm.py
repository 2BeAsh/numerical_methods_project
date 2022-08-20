#%% Imports
import numpy as np
import scipy as sp
import random
from scipy import sparse
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

#%% Getting rid of the for loop -
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

#%% Velocity Force
def force_vel(t, coord, a, b, c, p, L, eta, r0):
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

    # Set terms with distance 0 equal to 0 to avoid division by 0
    div_term_x = np.divide(xj - xk, dist_xy_p2, out=np.zeros((N,N), dtype=float), where=dist_xy_p2!=0) # Return zero where the denominator is 0
    div_term_y = np.divide(yj - yk, dist_xy_p2, out=np.zeros((N,N), dtype=float), where=dist_xy_p2!=0)

    fx = 1 / N * np.sum(div_term_x - a * (xj - xk), axis=1) \
                  + b * (x - zx) / dist_prey_pred

    fy = 1 / N * np.sum(div_term_y - a * (yj - yk), axis=1) \
                  + b * (y - zy) / dist_prey_pred

    fzx = c / N * np.sum((x - zx) / dist_prey_pred**(p/2)) # Divide p by 2 because normally squared
    fzy = c / N * np.sum((y - zy) / dist_prey_pred**(p/2))


    # Align angle according to average of neighbours' 
    
    # Calculate current angle, create single pos vector and use KDTree to find neighbours.
    # Then apply rotation matrix with the angle = new angle - old angle
    theta = np.arctan(fy/fx) 
    pos = np.empty((N,2))
    pos[:, 0] = x
    pos[:, 1] = y
    tree = KDTree(pos, boxsize=[L+0.01,L+0.01])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0, output_type="coo_matrix")
    data = np.exp(theta[dist.col]*1j)
    neigh = sparse.coo_matrix((data, (dist.row, dist.col)), shape=dist.get_shape())
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))

    theta_new = np.angle(S) + eta * np.random.uniform(-np.pi, np.pi, size=N)
    theta_rot = theta_new - theta
    fx_rot = fx * np.cos(theta_rot) - fy * np.sin(theta_rot)
    fy_rot = fx * np.sin(theta_rot) + fy * np.cos(theta_rot)

    return fx_rot, fy_rot, fzx, fzy


#%% Create movement of prey and predator

    
def movement(N, L, t_end, dt, a, b, c, p, boundary, r_eat, eta, r0):
    """
    N: int - Amount of prey
    L: float - Starting area which prey can be in
    t_end: float - How long the simulation should run
    dt: time step between each update
    """
    # Initial values for prey and predator
    # Prey are uniformly distributed in a square with side L
    t_vals = np.arange(0, t_end, dt)
    x = np.random.uniform(low=0, high=L, size=N)
    y = np.random.uniform(low=0, high=L, size=N)
    fx = np.random.uniform(-1, 1,size=N) 
    fy = np.random.uniform(-1, 1,size=N) 

    zx = 0.8 * L # Predator starts slightly south west of center
    zy = 0.2 * L
    x_list = [x]
    y_list = [y]
    fx_list = [fx]
    fy_list = [fy]
    zx_list = [zx]
    zy_list = [zy]

    # Predator starts with 0 velocity
    fzx_list = [0]
    fzy_list = [0]

    N_list = [N] # Amount of prey. Used to track how many preys eaten

    for t in t_vals:
        # Get and update values
        prey_coord = [x, y]
        predator_coord = [zx, zy]
        fx, fy, fzx, fzy = force_vel(t=t, coord=(prey_coord, predator_coord), a=a, b=b, c=c, p=p, L=L, eta=eta, r0=r0)
        x = x + fx * dt
        y = y + fy * dt
        zx = zx + fzx * dt
        zy = zy + fzy * dt
        
        if boundary == "stop":
            # Stop particles and set their speed equal to zero at boundaries
            fx[x < 0.001 * L] = -fx[x < 0.001 * L]
            fx[x >= 0.99 * L] = -fx[x >= 0.99 * L]
            fy[y < 0.001 * L] = -fy[y < 0.001 * L]
            fy[y >= 0.99 * L] = -fy[y >= 0.99 * L]
            fzx = np.where(zx>0.99*L or fzx>0.001*L, 0, fzx) # Able to use "or" because fzx is float and not array like fx
            fzy = np.where(zy>0.99*L or fzy>-0.001*L, 0, fzy)
    
            # Create square boundary which the particles cannot escape
            x = np.clip(x, 0, L)
            y = np.clip(y, 0, L)
            zx = np.clip(zx, 0, L)
            zy = np.clip(zy, 0, L)


        if boundary == "periodic":
            L_vec = np.ones(x.size)
            # Prey
            x = np.where(x>0.99*L, np.minimum(x-L, L_vec*0.99*L), x) # Top border
            x = np.where(x<0.01*L, np.maximum(L-x, L_vec*0.01*L), x) # bottom border
            y = np.where(y>0.99*L, np.minimum(y-L, L_vec*0.99*L), y) 
            y = np.where(y<0.01*L, np.maximum(L-y, L_vec*0.01*L), y) 
            # Predator
            zx = np.where(zx>0.99*L, np.minimum(zx-L, 0.99*L), zx) 
            zx = np.where(zx<0.01*L, np.maximum(L-zx, 0.01*L), zx) 
            zy = np.where(zy>0.99*L, np.minimum(zy-L, 0.99*L), zy) 
            zy = np.where(zy<0.01*L, np.maximum(L-zy, 0.01*L), zy) 

        accepted_boundary = ["stop", "periodic"]
        if boundary not in accepted_boundary:
            print("The boundary has no support, and no boundary is enforced!")

        # Eat prey by calculating distance from prey to pred and comparing to eat radius, and then removing prey inside eat radius
        r = np.sqrt((x - zx)**2 + (y - zy)**2)
        x = x[r > r_eat]
        y = y[r > r_eat]
        fx = fx[r > r_eat]
        fy = fy[r > r_eat]

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
        if N_list[-1] == 0:
            return x_list, y_list, zx_list, zy_list, fx_list, fy_list, fzx_list, fzy_list, N_list

        
    return x_list, y_list, zx_list, zy_list, fx_list, fy_list, fzx_list, fzy_list, N_list
#%%

#%% Animation function
def ani_func(N, L, t_end, dt, a, b, c, p, r_eat, eta, r0, boundary="stop"):
    # Set up figure and axis
    fig, ax = plt.subplots(dpi=125)
    ax.set(xlim=(0, L), ylim=(0, L))

    # Get data from movement function
    x_list, y_list, zx_list, zy_list, fx_list, fy_list, fzx_list, fzy_list, N_list = movement(N, L, t_end, dt, a, b, c, p, boundary, r_eat, eta, r0)
    x_len_start = len(x_list[0])
    max_vel_list_x = np.empty(len(x_list))
 
    
    print("Jeg animerer lige nu. Vent et øjeblik")
    
    # Fill in NANs
    for i in range(len(x_list)):
        if len(x_list[i]) < x_len_start:
            x_len_diff = x_len_start - len(x_list[i])
            max_velocity_x = sorted(fx_list[i])[70]

            
           
            a = [np.nan] * x_len_diff
                        
            x_list[i] = x_list[i].tolist()
            y_list[i] = y_list[i].tolist()
            fx_list[i] = fx_list[i].tolist()
            fy_list[i] = fy_list[i].tolist()

            
            x_list[i].extend(a)
            y_list[i].extend(a)
            fx_list[i].extend(a)
            fy_list[i].extend(a)
    
    max_velocity_x = sorted(max_vel_list_x)[-10]

    
    fx_list = np.clip(fx_list, -1, 1)
    fy_list = np.clip(fy_list, -1, 1)
   
            
    # First line
    scat_prey = ax.scatter(x_list[0], y_list[0], color="b", s=3)
    scat_pred = ax.scatter(zx_list[0], zy_list[0], color="r")
    quiv = ax.quiver(x_list[0], y_list[0], fx_list[0], fy_list[0] )
    
    
    # Boundary Box
    ax.plot([0, 0, L, L, 0], [0, L, L, 0, 0], linewidth=1.5, color="darkgreen") 
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axis('off')
  

    # Update line function
    def animation(i):
        # Update scatter
        scat_prey.set_offsets(np.c_[x_list[i], y_list[i]])
        scat_pred.set_offsets(np.c_[zx_list[i], zy_list[i]])

        # Update quiver
        quiv.set_offsets(np.c_[x_list[i], y_list[i]])
        quiv.set_UVC(fx_list[i], fy_list[i]) 

        # Update labels
        prey_eat = np.count_nonzero(np.isnan(x_list[i]))
        prey = np.count_nonzero(~np.isnan(x_list[i]))
        ax.set_title(f'Alive prey: {prey}, Dead prey: {prey_eat}, Time: {np.round(i * dt, 2)}', loc='left')
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_axis_off()
    anim = FuncAnimation(fig, animation, interval= t_end , frames=len(x_list))
    anim.save("animation_3.mp4")
    #plt.draw()
    #plt.show()

#%% Test animation function
ani_func(N=300, L=2, t_end=20, dt=0.01, a=1, b=0.7, c=6, p=2.7, r_eat=0.02, eta=0.015, r0=0.05, boundary="stop")

#%%
def plot_quiver():
    # Normalize vector fx, fy...
    x, y, zx, zy, fx, fy, fzx, fzy, N_living =  movement(N=1, L=1, t_end=2, dt=0.01, a=1, b=1, c=1, p=2.5, boundary="periodic")
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
        plt.xlim(0,1)
        plt.ylim(0,1)
#plot_quiver()