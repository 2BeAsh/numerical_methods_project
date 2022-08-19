import numpy as np
import matplotlib.pyplot as plt



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

xx = np.arange(10)
force_no_loop(t=0, coord = ((xx,xx),(0,0)), a=1, b=1, c=1, p=1)