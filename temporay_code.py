import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%%
c = ([0.46379, 1.03103, 1.44482, 1.25793, 1.371379, 1.4848,
1.5982, 1.71172, 1.8251, 1.9386, 2.0520, 2.1655, 2.27889, 2.3924,
2.50586, 2.6193, 2.7327, 2.8462, 2.9596, 3.0731, 3.18655] )


x = sorted([1.729,  2.1792, 1.6889, 1.62079, 1.3095, 1.262, 1.746,
1.3259, 1.5138, 1.4457, 1.8287, 1.7889, 2.0977, 1.88527, 2.0491,
2.33931, 1.7312, 1.8856, 2.0397, 2.20169, 2.442] )
plt.figure(figsize=(8, 6))
plt.plot(c, x, ".")
plt.scatter(c,x, color="black")
plt.show()
#%%

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

    return xj-xk
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


def force_no_loop_combined(t, prey_coord, pred_coord, a, b, c, p):
    x, y = prey_coord
    zx, zy = pred_coord
    N = x.size

    xk = x[:, None]
    xj = x[None, :]

    yk = y[:, None]
    yj =y[None, :]

    dist_prey_pred_p = np.sqrt((x - zx)**2 + (y - zy)**2)

    dist_xy_p2 = (xj - xk)**2 + (yj - yk)**2
    div_term_x = np.where(dist_xy_p2 == 0, 0, (xj - xk) / dist_xy_p2) # Set terms with distance = 0 to 0
    div_term_y = np.where(dist_xy_p2 == 0, 0, (yj - yk) / dist_xy_p2)

    fx = 1 / N * (np.sum(div_term_x - a * (xj - xk), axis=1)
                  + b * (x - zx) / dist_prey_pred_p**2)

    fy = 1 / N * (np.sum(div_term_y - a * (yj - yk), axis=1)
                  + b * (y - zy) / dist_prey_pred_p**2)

    fzx = c / N * np.sum((x - zx) / dist_prey_pred_p**p)
    fzy = c / N * np.sum((y - zy) / dist_prey_pred_p**p)
    return fx, fy, fzx, fzy

#%%
x = np.arange(15)
y = np.array([2,3,5,6,7,8,9,1,10,2])
#print(x)
#print(y)
#print(force_no_loop_combined(t=1, prey_coord=(x,y), pred_coord=(1,1), a=1,b=1,c=1,p=1))


#%%
import scipy as sp
from scipy import sparse
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
#%%

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
        Each prey particle??s derivative in the x-direction
    fy : 1darray
        Each prey particle??s derivative in the y-direction
    fzx: 1darray
        The predator??s derivative in the x-direction
    fzy: 1darray
        The predator??s derivative in the y-direction

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

#%% Vores tilf??lde
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
    
    d = 2

    # Get x and y specific, set terms with distance 0 equal to 0
    div_term_x = d*np.divide(xj - xk, dist_xy_p2, out=np.zeros((N,N), dtype=float), where=dist_xy_p2!=0)
    div_term_y = d*np.divide(yj - yk, dist_xy_p2, out=np.zeros((N,N), dtype=float), where=dist_xy_p2!=0)

    #div_term_x = np.where(dist_xy_p2 == 0, 0, (xj - xk) / dist_xy_p2)
    #div_term_y = np.where(dist_xy_p2 == 0, 0, (yj - yk) / dist_xy_p2)

    fx = 1 / N * np.sum(div_term_x - a * (xj - xk), axis=1) \
                  + b * (x - zx) / (dist_prey_pred+1e-8)

    fy = 1 / N * np.sum(div_term_y - a * (yj - yk), axis=1) \
                  + b * (y - zy) / dist_prey_pred

    fzx = c / N * np.sum((x - zx) / dist_prey_pred**(p/2)) # Divide p by 2 because normally squared
    fzy = c / N * np.sum((y - zy) / dist_prey_pred**(p/2))

    
    epsilon = 0.01
        
    theta = np.arctan(fy/fx)
    pos = np.empty((N,2))
    pos[:, 0] = x
    pos[:, 1] = y
    pos = np.clip(pos, 0+epsilon, 0.99*L - epsilon)
    tree = KDTree(pos, boxsize=[L+1,L+1])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0, output_type="coo_matrix")
    data = np.exp(theta[dist.col]*1j)
    neigh = sparse.coo_matrix((data, (dist.row, dist.col)), shape=dist.get_shape())
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))

    theta_ny = np.angle(S) + eta * np.random.uniform(-np.pi, np.pi, size=N)
    theta_rot = theta_ny - theta
    fx_rot = fx * np.cos(theta_rot) - fy * np.sin(theta_rot)
    fy_rot = fx * np.sin(theta_rot) + fy * np.cos(theta_rot)

    return fx_rot, fy_rot, fzx, fzy
 


#%% Create movement of prey and predator
def movement(N, L, t_end, dt, a, b, c, p, eta, r0):
    """
    N: int - Amount of prey
    L: float - Starting area which prey can be in
    t_end: float - How long the simulation should run
    dt: time step between each update
    """
    # Initial values for prey and then predator
    # Prey are uniformly distributed in a square with side L
    t_vals = np.arange(0, t_end, dt)
    x = np.random.uniform(low=0, high=L, size=N)
    y = np.random.uniform(low=0, high=L, size=N)
    fx = np.random.random(size=N)
    fy = np.random.random(size=N)

    zx = L/2 + 0.2
    zy = L/2 + +.2
    x_list = [x]
    y_list = [y]

    fx_list = [fx]
    fy_list = [fy]

    # Predator starts in the middle of square
    zx_list = [zx]
    zy_list = [zy]

    # Predator starts with zero velocity
    fzx_list = [0]
    fzy_list = [0]

    N_list = [N] # Amount of prey

    for t in t_vals:
        # Get values
        prey_coord = [x, y]
        predator_coord = [zx, zy]
        fx, fy, fzx, fzy = force_vel(t=t, coord=(prey_coord, predator_coord), a=a, b=b, c=c, p=p, L=L, eta=eta, r0=r0)
        # Update values
        x = x + fx * dt
        y = y + fy * dt
        zx = zx + fzx * dt
        zy = zy + fzy * dt

        # Set speed equal to zero at boundaries
        # THIS MIGHT BE EASIER - fx = np.where(x > 0.99 or x<-0.99, 0, x)
        fx[x<=0] = -fx[x<=0]
        fx[x>L*0.99] = -fx[x>L*0.99]
        fy[y<=0] = -fy[y<=0]
        fy[y>L*0.99] = -fy[y>L*0.99]
        fzx = np.where(zx>0.99*L or zx>0.01, 0, fzx) # Able to use "or" because fzx is float and not array like fx
        fzy = np.where(zy>0.99*L or zy>0.01, 0, fzy)

        # Create square boundary which the particles cannot escape
        x = np.clip(x, 0, 0.99*L)
        y = np.clip(y, 0, 0.99*L)
        zx = np.clip(zx, 0, 0.99*L)
        zy = np.clip(zy, 0, 0.99*L)


        # REMEMBER TO ADD PREDATOR HERE

        # Eat prey by calculating distance from prey to pred and comparing to eat radius, and then removing prey inside eat radius
        r_min = 0.05 # Predator eats prey within this radius
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


#%% Animation function
def ani_func(prey_list, pred_list, prey_deriv_list, pred_deriv_list, dt, L=1):
    # Set up figure and axis
    fig, ax = plt.subplots(dpi=125)
    ax.set(xlim=(0, 2), ylim=(0, 2))

    # Get data
    x_list, y_list = prey_list
    zx_list, zy_list = pred_list
    fx_list, fy_list = prey_deriv_list
    fzx_list, fzy_list = pred_deriv_list

    # First line
    scat_prey = ax.scatter(x_list[0], y_list[0], color="b", s=2)
    scat_pred = ax.scatter(zx_list[0], zy_list[0], color="r")
    #quiv = ax.quiver(x_list[0], y_list[0], fx_list[0], fy_list[0]) # Documentation says quiver([X, Y], U, V), but we have quiver(X, Y, U, V)

    # Boundary Box
    ax.plot([-L, -L, L, L, -L], [-L, L, L, -L, -L], "k--")

    # Labels
    label_eat = ax.text(1.9, 1.6, "Prey Eaten: 0", ha="right", va="center", fontsize=12)
    label_time = ax.text(1.9, 1.3, "Time: 0", ha="right", va="center", fontsize=12)

    # Update line function
    def animation(i):
        # Update scatter
        scat_prey.set_offsets(np.c_[x_list[i], y_list[i]])
        scat_pred.set_offsets(np.c_[zx_list[i], zy_list[i]])

        # Update quiver
       # print("x:", x_list[i].size)
        #print("y:", y_list[i].size)
        #print("fx:", fx_list[i].size)
        #print("fy:", fy_list[i].size)
        #quiv.set_offsets(np.c_[x_list[i], y_list[i]]) #
        #quiv.set_UVC(fx_list[i], fy_list[i]) # Has a problem when prey are eaten

        # Update labels
        prey_eat = len(x_list[0]) - len(x_list[i])
        time_count_str = "Time: " + str(round(i * dt, 1))
        label_eat.set_text(f"Prey Eaten: {prey_eat}")
        label_time.set_text(time_count_str)

    anim = FuncAnimation(fig, animation, interval=500, frames=len(x_list))
    print("worked")
    anim.save("animation_test.mp4")
    #plt.draw()
    #plt.show()
#%%
N=400
prey_coord = (np.random.uniform(0, 0.99, size=N), np.random.uniform(0, 0.99, size=N))
pred_coord = (0.11, 0.11)
#fx, fy, fzx, fzy = force_vel(t=1, coord=(prey_coord, pred_coord), a=1, b=1, c=2.2, p=3, L=1, eta=0.1, r0=0.1)

L = 2.5
x, y, zx, zy, fx, fy, fzx, fzy, N_living =  movement(N=N, L=L, t_end=1, dt=0.02, a=1, b=0.2, c=2.8, p=3, eta = 0.05, r0 = 0.06)
#ani_func((x, y), (zx, zy), (fx, fy), (fzx, fzy), dt=0.2)

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
    plt.scatter(x[i], y[i], s=6)
    plt.scatter(zx[i], zy[i], color="r", s=9)
    plt.quiver(x[i], y[i], fx[i] , fy[i])
    plt.quiver(zx[i], zy[i], fzx[i] , fzy[i], color="r")
    plt.xlim(0,L)
    plt.ylim(0,L)