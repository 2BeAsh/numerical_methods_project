import numpy as np
import matplotlib.pyplot as plt



def force_no_loop(t, coord, a, b, c, p):
    x, y = coord
    N = x.size
    
    xk = x[:, None]
    xj = x[None, :]
    xjk = xj - xk
    
    yk = y[:, None]
    yj =y[None, :]

    
    dist_xy_p2 = np.sum((xj - yj)**2 + (xk - yk)**2, axis=1)
    #dist_xz_p2 = np.sum((xj - zj)**2 + (xj - zj)**2, axis=1)
    
    fx = 1 / N * (np.sum(xj - xk, axis=1) / dist_xy_p2 - a * (np.sum(xj - xk, axis=1)))
    fy = 1 / N * (np.sum(yj - yk, axis=1) / dist_xy_p2 - a * (np.sum(yj - yk, axis=1)))
    return fx, fy

x = np.arange(10)
print(force_no_loop(t=1, coord=(x,x), a=1,b=1,c=1,p=1))