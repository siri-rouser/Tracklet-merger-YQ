import numpy as np
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt

# Data
data = np.array([1, 0.9, 1.1, 1, 1]).reshape(-1, 1)

# Kernel Density Estimation
kde = KernelDensity(kernel='gaussian', bandwidth=1.5).fit(data)
x_d = np.linspace(-10, 10, 1000).reshape(-1, 1)
log_dens = kde.score_samples(x_d)


test_value = np.array([[6]])
print('Test Value:', np.exp(kde.score(test_value)))

# Visualization
plt.fill_between(x_d[:, 0], np.exp(log_dens), alpha=0.5)
plt.plot(data[:, 0], np.full_like(data[:, 0], -0.01), '|k', markeredgewidth=1)
plt.title('Kernel Density Estimation')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()