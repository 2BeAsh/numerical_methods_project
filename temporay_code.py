import numpy as np
import matplotlib.pyplot as plt



def force_no_loop(t, prey_coord, pred_coord, a, b, c, p):    
    x, y = prey_coord
    zx, zy = pred_coord
    N = x.size
    
    xk = x[:, None]
    xj = x[None, :]

    yk = y[:, None]
    yj =y[None, :]

    dist_xy_p2 = np.sum((xj - xk)**2 + (yj - yk)**2, axis=1)
    dist_prey_pred_p = np.sqrt((x - zx)**2 + (y - zy)**2)
    
    fx = 1 / N * (np.sum(xj - xk, axis=1) / dist_xy_p2 
                  - a * np.sum(xj - xk, axis=1) 
                  + b * (x - zx) / dist_prey_pred_p**2)
    
    fy = 1 / N * (np.sum(yj - yk, axis=1) / dist_xy_p2
                  - a * np.sum(yj - yk, axis=1)
                  + b * (y - zy) / dist_prey_pred_p**2)
    
    fzx = c / N * np.sum((x - zx) / dist_prey_pred_p**p)
    fzy = c / N * np.sum((y - zy) / dist_prey_pred_p**p)
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
y = np.array([2,3,5,6,7,8,9,1,10,2,1,1,1,2,3])
#print(x)
#print(y)
print(force_no_loop_combined(t=1, prey_coord=(x,y), pred_coord=(1,1), a=1,b=1,c=1,p=1))

