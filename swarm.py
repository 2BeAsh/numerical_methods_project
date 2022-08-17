#%% Imports
import numpy as np
import matplotlib.pyplot as plt

#%% Create Swarm
def swarm(N, L):
    """
    N: int - Amount of prey
    L: float - Starting area which prey can be in
    """
    #Create uniformly distributed random positions
    x = np.random.uniform(low=-L, high=L, size=(N, 2))
    

    return x


print(swarm(N=5))
