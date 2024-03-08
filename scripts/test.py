import numpy as np
from scipy.optimize import minimize

# Given values
y = np.random.randn(256, 1)  # Example data
x = np.random.randn(1024, 1)  # Example estimate of x
lambda_reg = 0.1  # Regularization parameter

# Objective function to minimize
def objective_func(A_flat):
    A = A_flat.reshape(256, 1024)
    return np.linalg.norm(y - A @ x)**2 + lambda_reg * np.linalg.norm(A)**2

# Initial guess for A
A_init = np.random.randn(256, 1024).flatten()

# Optimization
result = minimize(objective_func, A_init, method='L-BFGS-B')

# Reshape A back to its original shape
A_opt = result.x.reshape(256, 1024)