from scipy.linalg import cholesky
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class UKF2D():
    """
        Class for Unscented Kalman Filter in 2 dimensions
    """
    def __init__(
        self,
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

        self.kappa = kappa # choose kappa >= 0
        self.alpha = alpha # choose alpha \in (0,1]
        self.beta = beta # choose Beta as 2 for normal distributions

        self.lmbd = self.alpha**2 * ( self.n + self.kappa ) - self.n
        
        self.mu = mu
        self.sigma = sigma
        
        self.plot_bool = plot_bool
        
        # self.z = z
        self.g = g
        self.h = h
        
    
    def plot_cov(m, C, sigma_points=None):
        fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(6,6))

        axs.axvline(c='black', alpha=0.5, linestyle='--')
        axs.axhline(c='black', alpha=0.5, linestyle='--')

        m = np.zeros(2)
        C = np.eye(2)

        lambda_, v = np.linalg.eig(C)
        lambda_ = np.sqrt(lambda_)

        for j in range(1,3):
            elli = Ellipse(xy=(m[0], m[1]),
                        width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                        angle=np.rad2deg(np.arccos(v[0, 0])))
            elli.set_facecolor('none')
            elli.set_edgecolor('red')
            axs.add_artist(elli)

        samples = np.random.multivariate_normal(mean=m, cov=C, size=1000)
        axs.scatter(samples[:,0], samples[:,1], alpha=0.5)

        if sigma_points is not None:
            axs.scatter(sigma_points[:,0], sigma_points[:,1], c='green', s=50, marker='x')
        
        #elli = Ellipse([1,1], width=1, height=1, angle=0)
        axs.set_xlim([m[0] - C[0,0] * 5,m[0] + C[0,0] * 5])
        axs.set_ylim([m[0] - C[1,1] * 5,m[0] + C[1,1] * 5])
        axs.add_patch(elli)
        plt.plot()
        
    
    def calc_sigma_points(self, mu, sigma):
        # 1: calculate sigma points: define chi_{t-1}
        pd_sigmas = self.approx_pos_definite((self.n + self.lmbd) * sigma)
        print(pd_sigmas)
        sigma_point_shift = cholesky(pd_sigmas)  #np.sqrt( (self.n + self.lmbd) * sigma )
        chi = np.zeros([self.n*2 + 1, self.n])
        chi[0] = mu
        for i in range(1, self.n + 1):
            #                                            Take the column vector
            chi[i] = np.subtract(mu, -sigma_point_shift[:, i - 1]) #np.sqrt( (self.n + self.lmbd) * sigma )[:, i - 1]
        for i in range(self.n + 1, 2*self.n + 1):
            #                                            Take the column vector
            chi[i] = np.subtract(mu, sigma_point_shift[:, i - self.n - 1]) #np.sqrt( (self.n + self.lmbd) * sigma )[:, i - self.n - 1]
        return chi
        
    def is_pos_definite(self, P):
        try: 
            _ = cholesky(P)
            return True
        except np.linalg.LinAlgError:
            return False
        
    def approx_pos_definite(self, A):
        # assert to get a square matrix
        assert(A.shape[0] == A.shape[1])

        if self.is_pos_definite(A):
            return A 

        B = (A + A.T) / 2

        U, Sigma, V = np.linalg.svd(A)
        H = V @ Sigma @ V

        A_hat = (B + H) / 2

        A_hat = (A_hat + A_hat.T) / 2

        # The original algorithm uses the eps function which is described in Matlab as 
        # eps returns the distance from 1.0 to the next larger double-precision number, that is, 2-52.
        # np.spacing can be considered as a generalization of EPS
        eps = np.spacing(np.linalg.norm(A))

        
        # I = np.eye(A.shape[0])


        k = 0

        while not self.is_pos_definite(A_hat):
            
            k += 1

            # np.linalg.eigvals can return complex numbers
            # Therefore we transform the value to real values first
            lambdas = np.real(np.linalg.eigvals(A_hat))
            min_lambda = np.min(lambdas)
            
            # The original algorithm uses the eps function which is described in Matlab as 
            # eps returns the distance from 1.0 to the next larger double-precision number, that is, 2-52.
            # np.spacing can be considered as a generalization of EPS
            # eps = np.spacing(min_lambda)


            A_hat = A_hat + np.eye(A.shape[0]) * ( -min_lambda * k**2 + eps)

        return A_hat
        
    def UKF(self, u, z):
        chi_prev = self.calc_sigma_points(self.mu, self.sigma)
        
        # TODO REMOVE
        self.chi_prev = chi_prev

        if self.plot_bool:        
            self.plot_cov(np.zeros(2), np.eye(2), chi_prev)





        # calculate weights
        self.ws = np.ones((2*self.n + 1, 2))
        self.ws[0,0] = self.lmbd / (self.n + self.lmbd)
        self.ws[0,1] = self.ws[0,0] + ( 1 - self.alpha**2 + self.beta )



        for i in range(1, 2*self.n + 1):
            self.ws[i,0] = 1/(2*(self.n + self.lmbd))
            self.ws[i,1] = self.ws[i,0]

        # 2: apply non linear map from motion model
        chi_bar_star = self.g(u, chi_prev) #g(u, chi_prev)

        self.chi_bar_star = chi_bar_star
        
        # STEP 3 calculate mu_bar | x_prior in other implementation
        mu_bar = self.ws[:,0] @ chi_bar_star

        self.mu_bar = mu_bar
        
        #for i in range(2*n):
        #    w[c]
    #     mu_bar = 0
    #     for i, w in enumerate(ws[:,0]):
    #         mu_bar += w * chi_bar_star[i]

        #mu_bar = ws[:,0] * chi_bar_star[:] #np.sum( ws[:,0] @ chi_bar_star[:], axis=0 ) #np.dot(ws[:,0], chi_bar_star[:]) )

        # STEP 4 calculate sigma_bar
        # TODO: MOVE to __init__
        R = np.eye(self.n) * 0.5

        y = chi_bar_star - mu_bar[np.newaxis, :]
        
        #print(f'y: {y}')
        sigma_bar = np.dot(y.T, np.dot(np.diag(self.ws[:,1]), y)) + R

        self.sigma_bar = sigma_bar
        
        # This is not working. Apparently. Code above works as expected
    #     sigma_bar = np.zeros([2,2])
    #     for i in range(2*n+1):
    #         sigma_bar += ws[i,1] * (chi_bar_star[i] - mu_bar) @ (chi_bar_star[i] - mu_bar).T + R

        #sigma = ws[:,1] @ ( chi_bar_star - mu_bar ) @ ( chi_bar_star - mu_bar ).T + R

        # UPDATE 
        # STEP 5: calculate chi_bar

        chi_bar = self.calc_sigma_points(mu_bar, sigma_bar)
        # TODO REMOVE
        self.chi_bar = chi_bar

        # STEP 6: Apply non linear map from observations
        #h = lambda x: x + 0.1
        Z_bar = self.h(chi_bar)
        
        self.Z_bar = Z_bar

        # STEP 7: calculate z_hat
        z_hat = self.ws[:,0].reshape(1,-1) @ Z_bar

        # TODO REMOVE
        self.z_hat = z_hat



        # STEP 8: Calculate Std_dev 
        Q = np.eye(self.n) * 0.5
        y = Z_bar - z_hat
        S_pred = np.dot(y.T, np.dot(np.diag(self.ws[:,1]), y)) + Q    #np.dot(y.T, np.dot(np.diag(Wc), y))
        
        self.y = y.copy()
        self.S_pred = S_pred

    #     Q = np.array([[0.5,0],[0,0.5]])
    #     Std_dev = np.zeros([2,2])
    #     for i in range(2*n):
    #         Std_dev += ws[i,1] * (Z_bar[i] - z_hat) @ (Z_bar[i] - z_hat).T + Q

        # STEP 9: Cross Variance

        # Sigma_bar_cross = np.zeros((chi_bar.shape[1], Z_bar.shape[1]))
        # N = chi_bar.shape[0]
        #pdb.set_trace()

        # print('np.subtract(chi_bar, mu_bar)\n', np.subtract(chi_bar, mu_bar))
        Sigma_bar_cross = ( ( np.subtract(chi_bar, mu_bar) ) * self.ws[:,1].reshape(-1, 1) ).T @ ( np.subtract(Z_bar, z_hat) )


        # for i in range(N):
        #     dx = np.subtract(chi_bar[i], mu_bar)
        #     dz = np.subtract(Z_bar[i], z_hat)
        #     Sigma_bar_cross += self.ws[:,1][i] * np.outer(dx, dz)


        self.Sigma_bar_cross = Sigma_bar_cross
    #     Sigma_pred = np.zeros([2,2])
    #     for i in range(2*n):
    #         Sigma_pred += ws[i,1] * (chi_bar[i] - mu_bar) @ (Z_bar[i] - z_hat).T

        # STEP 10: Calculate Kalman Gain
        self.KG = np.dot(Sigma_bar_cross, np.linalg.inv(S_pred)) #np.linalg.inv(Std_dev)

        # STEP 11: Calculate mu
        self.mu = (mu_bar.reshape(-1,1) + self.KG @ (z - z_hat).reshape(-1,1)).reshape(1,-1).flatten()

        # STEP 12: Calculate Covariance with observation
        self.sigma = sigma_bar - np.dot(self.KG, np.dot(S_pred, self.KG.T))

        self.sigma = self.approx_pos_definite(self.sigma)

        return self.mu, self.sigma
    