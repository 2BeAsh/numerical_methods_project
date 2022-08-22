#%% Imports
import numpy as np
import scipy as sp
import random
from scipy import sparse

from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.animation 
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks

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
def count_dead(N, L, t_end, dt, a, b, c, p, boundary, r_eat, eta, r0):
        
        dead_list=[]
        t_list = np.linspace(0, t_end, 2001)
        print('start med loops')
        
        for n in range(50):
            x_list, y_list, zx_list, zy_list, fx_list, fy_list, fzx_list, fzy_list, N_list = movement(N, L, t_end, dt, a, b, c, p, boundary, r_eat, eta, r0)
            x_len_start = len(x_list[0])
            for i in range(len(x_list)):
                dead = x_len_start-len(x_list[i])
                dead_list.append(dead)
                
        dead_list = np.array(dead_list) 
        
        dead_list = np.reshape(dead_list, (50, int(len(x_list))), order='A')
        
        dead_list = np.mean(np.array(dead_list), axis=0)  
        
        print('done med loops')
        
        return dead_list, t_list

dead_listc1, t_range1 = count_dead(N=300, L=2, t_end=40, dt=0.02, a=1, b=0.5, c=4, p=2.5, r_eat=0.02, eta=0.015, r0=0.05, boundary="stop")
dead_listc2, t_range2 = count_dead(N=300, L=2, t_end=40, dt=0.02, a=1, b=0.5, c=6, p=2.5, r_eat=0.02, eta=0.015, r0=0.05, boundary="stop")
dead_listc3, t_range3 = count_dead(N=300, L=2, t_end=40, dt=0.02, a=1, b=0.5, c=7, p=2.5, r_eat=0.02, eta=0.015, r0=0.05, boundary="stop")
dead_listc4, t_range4 = count_dead(N=300, L=2, t_end=40, dt=0.02, a=1, b=0.5, c=8, p=2.5, r_eat=0.02, eta=0.015, r0=0.05, boundary="stop")
dead_listc5, t_range5 = count_dead(N=300, L=2, t_end=40, dt=0.02, a=1, b=0.5, c=9, p=2.5, r_eat=0.02, eta=0.015, r0=0.05, boundary="stop")

#%%
plt.figure()
plt.step(t_range1,dead_listc1, label="c=4")
plt.step(t_range2,dead_listc2, label="c=6")
plt.step(t_range3,dead_listc3, label="c=7")
plt.step(t_range4,dead_listc4, label="c=8")
plt.step(t_range5,dead_listc5, label="c=9")
plt.title("Prey eaten over time for different c, averaged over 50 runs")
plt.xlabel('time')
plt.ylabel('Prey eaten')
plt.legend()
plt.show()
#%%
def where_dead(N, L, t_end, dt, a, b, c, p, boundary, r_eat, eta, r0):
        
        corner=0
        middle=0
        edge=0
        
        dead_list=[0]
        #t_list = np.linspace(0, t_end, 2001)
        print('start med loops')
        
        for n in range(20):
            x_list, y_list, zx_list, zy_list, fx_list, fy_list, fzx_list, fzy_list, N_list = movement(N, L, t_end, dt, a, b, c, p, boundary, r_eat, eta, r0)
            x_len_start = len(x_list[0])
            for i in range(len(x_list)):
                dead = x_len_start-len(x_list[i])
                
                if dead != dead_list[-1]:
                    if (zx_list[i]<0.5 and zy_list[i]<0.5) or (zx_list[i]<0.5 and zy_list[i]>(L-0.5)) or (zy_list[i]<0.5 and zx_list[i]>(L-0.5)) or (zx_list[i]>(L-0.5) and zy_list[i]>(L-0.5)):
                        corner+=(dead-dead_list[-1])
                    elif zx_list[i]<0.5 or zx_list[i]>(L-0.5) or zy_list[i]<0.5 or zy_list[i]>(L-0.5):
                        edge+=(dead-dead_list[-1])
                    else:
                        middle+=(dead-dead_list[-1])
                
                dead_list.append(dead)
            dead_list.append(0)
        return corner, middle, edge

#corner, middle, edge = where_dead(N=300, L=6, t_end=30, dt=0.02, a=1, b=0.5, c=9.5, p=2.5, r_eat=0.02, eta=0.015, r0=0.05, boundary="stop")

#%%

def plot_where_dead():
    L_list = np.linspace(1.2, 4, 37)
    corner_list=[]
    middle_list=[]
    edge_list=[]
    for L in L_list:
        corner, middle, edge = where_dead(N=200, L=L, t_end=30, dt=0.03, a=1, b=0.5, c=9.5, p=2.5, r_eat=0.02, eta=0.015, r0=0.05, boundary="stop")
        total = corner + middle + edge
        corner_list.append(corner/total)
        middle_list.append(middle/total)
        edge_list.append(edge/total)
    
    plt.figure(figsize=(7, 5))
    plt.plot(L_list**2/200, corner_list,  "o", label='Corner')
    plt.plot(L_list**2/200, corner_list, label='Corner')
    plt.plot(L_list**2/200, middle_list, "o", label='Middle')
    plt.plot(L_list**2/200, middle_list, label='Middle')
    plt.plot(L_list**2/200, edge_list,"o", label='Edge')
    plt.plot(L_list**2/200, edge_list, label='Edge')
    plt.legend()
    plt.title('Where do they die?')
    plt.xlabel('Prey density (number of preys per area) ')
    plt.ylabel('Percentage dead')
    plt.show()

plot_where_dead()
#%%
a= np.array([1,2,3, 1,2,3])
a = np.reshape(a, (2,3))
print(a)
a = np.mean(a, axis=0)
print(a)

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
    
    f = r"C:\Users\caird\Documents\Cairui\Python1\Numerical Methods\Week4 Project\numerical_methods_project\save_animations.mp4" 
    anim = FuncAnimation(fig, animation, interval= t_end , frames=len(x_list))
    #writervideo = matplotlib.animation.FFMpegWriter(fps=60)
    anim.save(f, writer="ffmpeg")
    

#%% Test animation function

ani_func(N=300, L=2, t_end=20, dt=0.01, a=1, b=0.7, c=6.7, p=2.7, r_eat=0.02, eta=0.015, r0=0.05, boundary="stop")
#%%
import matplotlib
print(matplotlib.__version__)


#%%
def pred_bane(N, L, t_end, dt, a, b, c_list, p, boundary, r_eat, eta, r0):
    for c in c_list:
        
        x_list, y_list, zx_list, zy_list, fx_list, fy_list, fzx_list, fzy_list, N_list = movement(N, L, t_end, dt, a, b, c, p, boundary, r_eat, eta, r0)
        plt.figure()
        plt.plot(zx_list[1700:], zy_list[1700:], ".")
        plt.xlim(1,6)
        plt.ylim(1,6)
        plt.show()


#pred_bane(N=300, L=6, t_end=30, dt=0.01, a=1, b=0.7, c_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 4,4.5, 5,5.5, 6,6.5,7 ], p=2.7, r_eat=0.02, eta=0.015, r0=0.05, boundary="stop")

#%%
def pred_bane_x(N, L, t_end, dt, a, b, c_list, p, boundary, r_eat, eta, r0):

    
    peak_list=[]
    minima_list=[]
    #c_before_bifurcation = np.linspace(0.01, 3.3, 30)
    #print("går i gang med første loop")
    """
    for c in c_before_bifurcation:
        #t_list = np.linspace(0, t_end*10, 3000, endpoint=False)
        t_list = np.linspace(0, t_end, 13000, endpoint=False)
        t = t_end
        x_list, y_list, zx_list, zy_list, fx_list, fy_list, fzx_list, fzy_list, N_list = movement(N=N, L=L, t_end=t, dt=dt, a=a, b=b, c=c, p=p, boundary=boundary, r_eat=r_eat, eta=eta, r0=r0)
        
        print(np.array(zx_list)[-3:])
        plt.figure()
        plt.plot(t_list[:], zx_list[:-1], ".", markersize="3", label=f'c={c}')
        plt.ylim(0,5)
        plt.xlim(0,t_end)
        plt.title('Predator movement (x-coordinate)')
        plt.legend()
        plt.xlabel('time/ s')
        plt.ylabel('x-coordinate')
        plt.show()
    """    
        
    print("færdig med første loop")
    
    for c in c_list:
        

        x_list, y_list, zx_list, zy_list, fx_list, fy_list, fzx_list, fzy_list, N_list = movement(N, L, t_end, dt, a, b, c, p, boundary, r_eat, eta, r0)
        zx_revers = np.array(zx_list[:-1])*-1
        
        minima,_= find_peaks(zx_revers, height=-5)
        #print(zx_revers)
        #print(minima)
        peaks, _ = find_peaks(zx_list[:-1], height=2)
        
        
        """
        plt.figure()
        plt.plot(t_list[:], zx_list[:-1], ".", markersize="3", label=f'c={c}')
        plt.plot(np.array(t_list)[peaks], np.array(zx_list)[:-1][peaks], "x", markersize="7", label='Maxima')
        plt.plot(np.array(t_list)[minima], np.array(zx_list)[:-1][minima], "x", markersize="7", label='Minima')
       
        plt.ylim(0,5)
        plt.xlim(0,t_end)
        plt.title('Predator movement (x-coordinate)')
        plt.legend()
        plt.xlabel('time/ s')
        plt.ylabel('x-coordinate')
        plt.show()
        """
        
        peak_list.append(np.array(zx_list)[:-1][peaks][5:])
        minima_list.append(np.array(zx_list)[:-1][minima][5:])
    
    print("går i gang med sidste loop")
    c = ([ 1.03103, 1.44482, 1.25793, 1.371379, 1.4848,
    1.5982, 1.71172, 1.8251, 1.9386, 2.0520, 2.1655, 2.27889, 2.3924,
    2.50586, 2.6193, 2.7327, 2.8462, 2.9596, 3.0731, 3.18655, 3.211, 3.2995] )


    x = sorted([ 2.1792, 1.6889, 1.62079, 1.3095, 1.362, 1.746,
    1.3259, 1.5138, 1.4457, 1.8287, 1.7889, 2.0977, 1.88527, 2.0491,
    2.33931, 1.7312, 1.8856, 2.0397, 2.20169, 2.442, 2.541, 2.665] )
    plt.figure(figsize=(8, 6))
    plt.scatter(c, x, color="black", s=6)
    for x, y, z in zip(c_list, peak_list, minima_list):
        plt.scatter([x] * len(y), y, color='black', s=6)
        plt.scatter([x] * len(z), z, color='black', s=6)
        plt.xlabel('c-value')
        plt.ylabel('x')
    plt.show()


#pred_bane_x(N=300, L=6, t_end=30, dt=0.02, a=1, b=0.7, c_list=np.linspace(3.35, 15, 150), p=2.7, r_eat=0.02, eta=0.015, r0=0.05, boundary="stop")

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

#%%
c = ([0.46379, 1.03103, 1.44482, 1.25793, 1.371379, 1.4848,
1.5982, 1.71172, 1.8251, 1.9386, 2.0520, 2.1655, 2.27889, 2.3924,
2.50586, 2.6193, 2.7327, 2.8462, 2.9596, 3.0731, 3.18655] )


x = ([1.729,  2.1792, 1.6889, 1.62079, 1.3095, 1.262, 1.746,
1.3259, 1.5138, 1.4457, 1.8287, 1.7889, 2.0977, 1.88527, 2.0491,
2.33931, 1.7312, 1.8856, 2.0397, 2.20169, 2.442] )

plt.plot(c, x)
