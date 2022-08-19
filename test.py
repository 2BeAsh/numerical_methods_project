# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %% Test quiver animation

# Det med quiver doc siger andet end jeg har brugt


def ani_test(xy_vals, deriv_xy, dt, L):
    # Set up figure and axis
    fig, ax = plt.subplots(dpi=125)

    # Get data
    x_list, y_list = xy_vals
    fx_list, fy_list = deriv_xy

    # Fill in NANs

    # First line
    scat_prey = ax.scatter(x_list[0], y_list[0], color="b", s=2)
    # Documentation says quiver([X, Y], U, V), but we have quiver(X, Y, U, V)
    quiv = ax.quiver(x_list[0], y_list[0], fx_list[0], fy_list[0])

    # Update line function
    def animation(i):
        # Update scatter
        scat_prey.set_offsets(np.c_[x_list[i], y_list[i]])
        
        # Update quiver
        print("x:", x_list[i].size)
        #print("y:", y_list[i].size)
        print("fx:", fx_list[i].size)
        #print("fy:", fy_list[i].size, flush=True)
        quiv.set_offsets(np.c_[x_list[i], y_list[i]])
        quiv.set_UVC(fx_list[i], fy_list[i])

    anim = FuncAnimation(fig, animation, interval=125, frames=len(x_list))
    anim.save("animation_test2.mp4")


L = 10
N = 20
x_list = []
y_list = []
dx_list = []
dy_list = []

N_s = np.arange(N, N-10, -1)
print(N_s)
for N_val in N_s:
    x_list.append(np.random.uniform(low=0, high=L, size=N_val)) # ti er antal frames
    y_list.append(np.random.uniform(low=0, high=L, size=N_val)) # ti er antal frames
    dx_list.append(np.random.uniform(low=0, high=L, size=N_val)) # ti er antal frames
    dy_list.append(np.random.uniform(low=0, high=L, size=N_val)) # ti er antal frames


ani_test(xy_vals=[x_list, y_list],
         deriv_xy=[dx_list, dy_list],
         dt=0.2,
         L=L)

#%%
x_test = np.random.uniform(size=(10,10))
print(np.where(x_test>0.5, True, False))


