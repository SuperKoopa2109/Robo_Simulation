import numpy as np 

class vel_motion_model():

    def __init__(
            self,
            time_step = 0.1
    ):
        self.time_step = time_step

    def  forward(self, x_curr, u, x_prev):
        x, y , theta = x_prev
        x_prime, y_prime, theta_prime = x_curr
        v, w = u
        delta_t = self.time_step

        nom = (x - x_prime) @ np.cos(theta) + ( y - y_prime ) @ np.sin(theta)
        denom = ( y - y_prime ) @ np.cos(theta) + (x - x_prime) @ np.sin(theta)
        mu = (1/2) * ( nom / denom )

        x_star = ( x + x_prime ) / 2 + mu * ( y - y_prime )
        y_star = ( y + y_prime ) / 2 + mu * ( x - x_prime )

        r_star = np.sqrt( ( x - x_star )**2 + ( y - y_star )**2 )
        delta_theta = np.arctan2( y_prime - y_star, x_prime - x_star ) - np.arctan2( y - y_star, x - x_star )

        v_hat = ( delta_theta / delta_t ) * r_star
        w_hat = ( delta_theta / delta_t )
        gamma_hat = ( theta_prime - theta ) / delta_t - w_hat

        alpha_1 = 1
        alpha_2 = 1
        alpha_3 = 1
        alpha_4 = 1
        alpha_5 = 1
        alpha_6 = 1

        # return probabilities for x, y, theta
        return ( self.prob_normal_dist( v - v_hat, alpha_1 * np.abs(v) + alpha_2 * np.abs(w) ),
                self.prob_normal_dist( w - w_hat, alpha_3 * np.abs(v) + alpha_4 * np.abs(w) ),
                self.prob_normal_dist( gamma_hat, alpha_5 * np.abs(v) + alpha_6 * np.abs(w) ) )
    
    def  prob_normal_dist(self, a, b):
        return 1 / np.sqrt( 2 * np.pi * b) * np.exp( - (1/2) * ((a**2)/b) )
    
class sample_motion_model_velocity():
    def __init__(
            self,
            sampler = np.random.normal,
            time_step = 0.1
    ):
        self.sampler = lambda mu, sigma: sampler( loc = mu, scale = sigma )
        self.time_step = time_step

    def forward(self, u, x_prev):
        x, y, theta = x_prev
        v, w = u 
        delta_t = self.time_step
        
        alpha_1 = 0.1
        alpha_2 = 0.1
        alpha_3 = 0.1
        alpha_4 = 0.1
        alpha_5 = 0.1
        alpha_6 = 0.1

        v_hat = v + self.sampler(alpha_1 * np.abs(v) + alpha_2 * np.abs(w), 1 )
        w_hat = w + self.sampler(alpha_3 * np.abs(v) + alpha_4 * np.abs(w), 1 )
        gamma_hat = self.sampler(alpha_5 * np.abs(v) + alpha_6 * np.abs(w), 1 )

        x_prime = x - (v_hat / w_hat) * np.sin(theta) + (v_hat/w_hat) * np.sin(theta + w_hat * delta_t)
        y_prime = y + (v_hat / w_hat) * np.cos(theta) - (v_hat/w_hat) * np.cos(theta + w_hat * delta_t)
        theta_prime = theta + w_hat * delta_t + gamma_hat * delta_t 

        return np.array([x_prime, y_prime, theta_prime])
