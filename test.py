#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import swarm as my_mod

#%% Test Graph
x = np.arange(0, 10, 1)
y = np.random.randn(x.size)

plt.plot(x, y, ".")
plt.show()

#%%

print("hej")


#%%

print(swarm.prey_prey_deriv(t=1, x_vec=(np.arange(10), np.arange(10)), a=1))