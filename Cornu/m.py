import numpy as np

samples = np.array([500.5, 500.3, 500.1, 500.3, 500.5, 500.3, 500.7, 500.0, 500.8, 499.9])

print(f'm = {np.mean(samples)}')
print(f'random error = {np.std(samples)}')
print(f'systematic error = {abs(500.0 - np.mean(samples))}')