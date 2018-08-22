from numpy.random import normal
import numpy as np
import matplotlib.pyplot as plt

storage = []

for i in range(0,1000000):
    storage.append(6500.4 + normal(0, np.sqrt(10)))

plt.hist(storage,1000)
plt.show()