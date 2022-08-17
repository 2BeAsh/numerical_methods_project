#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from swarm import prey_prey_deriv

#%% Test Graph
x = np.arange(0, 10, 1)
y = np.random.randn(x.size)

plt.plot(x, y, ".")
plt.show()

#%%

print("hej")


#%%

print(prey_prey_deriv(t=1, x_vec=(np.arange(10), np.arange(10)), a=1))