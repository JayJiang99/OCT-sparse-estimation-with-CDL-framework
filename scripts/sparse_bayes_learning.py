import numpy as np
import pymc as pm
from scipy.fftpack import dct, idct

class SparseBayesianLearningPoisson:
    def __init__(self, A, y, x_init):
        """
        Initialize the solver with the problem setup.
        
        Parameters:
        - A: The sensing matrix (128x1024).
        - y: The observed data vector (128x1).
        - x_init: The initial estimation of the signal x (1024x1).
        """
        self.A = A
        self.y = y.reshape((len(y),1))
        self.x_init = x_init.reshape((len(x_init),1))
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Builds the PyMC model for the compressed sensing problem.
        """
        model = pm.Model()
        with model:
            # Define the Poisson noise (delta) as a matrix of shape 128x1024
            # Since Poisson noise is typically non-negative, we use a non-negative normal prior for simplicity
            
            delta = pm.Normal('delta', mu=0, sigma=1, shape=self.A.shape, initval=np.zeros(self.A.shape))
            
            # Define the signal x (1024x1) with the initial estimation as a starting point
            x = pm.Normal('x', mu=0, sigma=1, shape=(self.A.shape[1], 1), initval=self.x_init)
            
            # Define the Gaussian noise n (128x1)
            sigma_n = pm.HalfNormal('sigma_n', sigma=1)
            n = pm.Normal('n', mu=0, sigma=sigma_n, shape=(self.A.shape[0], 1))
            
            # The sensing equation including A, delta, and n
            mu_y = pm.math.dot(self.A + delta, x) + n
            
            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal('Y_obs', mu=mu_y, sigma=sigma_n, observed=self.y)
            
        return model
        
    def infer_signal(self, draws=100, tune=50):
        """
        Performs inference to estimate the signal x.
        
        Parameters:
        - draws: Number of samples to draw. (Default: 1000)
        - tune: Number of iterations to tune. (Default: 500)
        
        Returns:
        - A trace of the sampled posterior distribution.
        """
        with self.model:
            trace = pm.sample(draws, tune=tune, return_inferencedata=True)
        return trace

# def main():
#     # Your main code that creates instances of CompressedSensingSolver,
#     # runs the model, or any other task that involves multiprocessing
#     # Example usage
#     # Assuming A, y, and x_init are already defined with the correct shapes
#     A = np.random.randn(128, 1024)  # Example sensing matrix
#     y = np.random.randn(128, 1)  # Example observed data
#     x_init = dct(np.random.randn(1024, 1), norm='ortho')  # Initial estimate of x

#     solver = SparseBayesianLearningPoisson(A, y, x_init)
#     trace = solver.infer_signal(draws=1000, tune=500)

# if __name__ == '__main__':
#     # The following function is necessary for multiprocessing on Windows
#     from multiprocessing import freeze_support
#     freeze_support()
    
#     # Call the main function or directly place your code here
#     main()


