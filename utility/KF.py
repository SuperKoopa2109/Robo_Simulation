from scipy.linalg import cholesky
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class KF2D():
    """
        Class for Kalman Filter in 2 dimensions
    """

    def __init__(
            self,
            x_init = np.zeros([2]),
            A = np.eye(2),
            B = np.eye(2),
            C = np.eye(2),
            R = np.eye(2),
            Q = np.eye(2)
    ):
        self.A = A 
        self.B = B 
        self.C = C
        self.R = R 
        self.Q = Q 

        self.mu = x_init 
        self.Sigma = np.eye(2)

        self.dim = self.mu.shape[0]

    def predict(self, u_t):
        x_a_bar = self.mu
        Sigma_a_prev = self.Sigma
        x_f_bar = self.A @ x_a_bar + self.B @ u_t
        Sigma_f_bar = self.A @ Sigma_a_prev @ self.A.T + self.R

        self.x_f_bar = x_f_bar 
        self.Sigma_f_bar = Sigma_f_bar 

    def filter(self):
        x_f_bar = self.x_f_bar 
        Sigma_f_bar = self.Sigma_f_bar
        K = Sigma_f_bar @ self.C @ np.linalg.inv( ( self.C @ Sigma_f_bar + self.Q ) )
        x_a_bar = x_f_bar + K @ ( z - self.C @ x_f_bar )
        Sigma_a = ( np.eye(self.dim) - K @ self.C ) @ Sigma_f_bar 

        self.mu = x_a_bar
        self.Sigma = Sigma_a

        return (self.mu, self.Sigma)

class UKF2D():
    """
        Class for Unscented Kalman Filter in 2 dimensions
    """
    def __init__(
        self,
        u, 
        g, 
        h,
        mu: object = np.zeros(2),
        sigma: object = np.eye(2),
        alpha: float = 0.3,
        beta: float = 2.0,
        kappa: int = 3.0,
        plot_bool: bool = False):

        
        # mu: mean of gaussian
        # sigma: covariance matrix of gaussian
        # u: control
        # z: measurement

        self.n = mu.shape[0]
        
        self.dim = mu.shape

        self.mu = mu
        self.sigma = sigma
        
        self.plot_bool = plot_bool
        
        self.u = u
        # self.z = z
        self.g = g
        self.h = h
        
        def forward(self, z, u):
            mu_hat = self.mu + self.u
            sigma_hat = self.sigma + R

            kg = sigma_hat @ C.T @ np.linalg.inv( C @ sigma_hat @ C.T + Q )

            mu = mu_hat + kg @ ( z - C @ mu_hat )
            sigma = ( np.eye(dim) - kg @ C ) @ sigma_hat

            self.mu = mu
            self.sigma = sigma