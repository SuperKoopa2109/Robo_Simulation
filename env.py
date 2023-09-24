import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

from utility.UKF import UKF2D
from utility.vel_motion_model import sample_motion_model_velocity

# from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

class RoboMap():
    """Class for whole map, which contains multiple Objects
        width: in m
        height: in m"""
    def __init__(self, width, height, plot_scale=0.5):
        self.width = width
        self.height = height
        self.objects = []
        self.plot_scale = plot_scale
        
    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def add_object(self, obj):
        self.objects.append(obj)
        
    def get_objects(self):
        return self.objects
    
    def add_robo(self, robo):
        # TODO: make it possible to add multiple robots? And return robo id? 
        self.robo = robo

    def update_robot(self, robo, idx = 0):
        self.robo = robo
    
    # source: https://web.archive.org/web/20111108065352/https://www.cs.mun.ca/~rod/2500/notes/numpy-arrays/numpy-arrays.html
    #
    # line segment intersection using vectors
    # see Computer Graphics by F.S. Hill
    #
    def perp(self, a ) :
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    # line segment a given by endpoints a1, a2
    # line segment b given by endpoints b1, b2
    # return 
    def get_intersect(self, a1,a2, b1,b2) :
        da = a2-a1
        db = b2-b1
        dp = a1-b1
        dap = self.perp(da)
        denom = np.dot( dap, db)
        num = np.dot( dap, dp )
        if denom == 0:
            return None
        intersec = (num / denom)*db + b1

        delta = 1e-3

        # check if calculated intersection is actually between the lines
        condx_a = min(a1[0], a2[0])-delta <= intersec[0] and max(a1[0], a2[0])+delta >= intersec[0] #within line segment a1_x-a2_x
        condx_b = min(b1[0], b2[0])-delta <= intersec[0] and max(b1[0], b2[0])+delta >= intersec[0] #within line segment b1_x-b2_x
        condy_a = min(a1[1], a2[1])-delta <= intersec[1] and max(a1[1], a2[1])+delta >= intersec[1] #within line segment a1_y-b1_y
        condy_b = min(b1[1], b2[1])-delta <= intersec[1] and max(b1[1], b2[1])+delta >= intersec[1] #within line segment a2_y-b2_y
        if not (condx_a and condy_a and condx_b and condy_b):
            intersec = None #line segments do not intercept i.e. interception is away from from the line segments

        return intersec
    
    def _draw_scatter(self, data, ax=None, plot_type='matplotlib'):
        if plot_type == 'matplotlib':
            if len(data.shape) == 1:
                ax.scatter([data[0]], [data[1]])
            else:
                ax.scatter(data[:, 0], data[:, 1])
        elif plot_type == 'dash':
            scatter_plot = None
            data = data.reshape(-1, 2)
            scatter_plot = go.Scatter(
                    x=data[:, 0],
                    y=data[:, 1]
                )
            # if len(data.shape) == 1:
            #     scatter_plot = go.Scatter(
            #         x=data[0].reshape(1),
            #         y=data[1].reshape(1)
            #     )
            # else:
            #     scatter_plot = go.Scatter(
            #         x=data[0, :],
            #         y=data[1, :]
            #     )
            return scatter_plot
    
    def _draw_line(self, ax, startp, endp, c='black', ls='-', plot_type='matplotlib'):
        if plot_type == 'matplotlib':
            ax.plot([startp[0], endp[0]], [startp[1], endp[1]], c=c, ls=ls)
        elif plot_type == 'dash':
            return None
        
    
    def draw_map(self, show_lidar=False, plot_type='matplotlib'):
        if plot_type == 'matplotlib':
            # function to plot the map using matplotlib
            
            # Create figure and axes
            fig, ax = plt.subplots(figsize = (self.width * self.plot_scale + 2, self.height * self.plot_scale + 2))

            # Add outlines of map as rectangle
            rect = patches.Rectangle((0, 0), self.width, self.height, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            
            # Add objects to map
            all_segments = []
            for obj in self.objects:
                all_segments.append(obj.get_points())
                rect = patches.Rectangle((obj.pos[0], obj.pos[1]), obj.shape[0], obj.shape[1], linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                
            all_segments = np.array(all_segments)
            all_segments = all_segments.reshape(all_segments.shape[0] * all_segments.shape[1], all_segments.shape[2], all_segments.shape[3])
                
            # Add robot to axis
            robo_file = 'media/robo.png'
            robo = image.imread(robo_file)
            #The OffsetBox is a simple container artist.
            #The child artists are meant to be drawn at a relative position to its #parent.
            imagebox = OffsetImage(robo, zoom = 0.5)
            #Annotation box for solar pv logo
            #Container for the imagebox referring to a specific position *xy*.
            ab = AnnotationBbox(imagebox, (self.robo.X, self.robo.Y), frameon = False)
            ax.add_artist(ab)
            
            # show lidar scans
            if show_lidar:
                total_scans = self.robo.ranges.shape[0]
                
                
                for laser in self.robo.get_lidar_ranges():
                    
                    startpos, endpos = laser
                    
                    all_intersecs = []
                    
                    ray_intersec = None
                    min_d = np.linalg.norm(endpos - startpos)
                    
                    for seg in all_segments:
                        
                        intersec = self.get_intersect(startpos, endpos, seg[0], seg[1])
                        if intersec is not None:
                            # get L2 length of vector
                            d = np.linalg.norm(intersec - startpos)
                            if d < min_d:
                                min_d = d
                                ray_intersec = intersec
                                
                    if ray_intersec is not None:
                        #print(f'ray_intersec {ray_intersec}')
                        #print(f'ray_intersec.shape {ray_intersec.shape}')
                        self._draw_scatter(ray_intersec, ax=ax, plot_type=plot_type)
                        #ax.scatter([ray_intersec[0]/2], [ray_intersec[1]/2])        
                        
                    #intersec = [self.get_intersect(startpos, endpos, seg[0], seg[1]) for seg in all_segments]
                    
                    self._draw_line(ax, startpos, endpos, ls='--', plot_type=plot_type)
                    #ax.plot([startx/2, endx/2], [starty/2, endy/2], c='black', ls='--')
            
            # set axis dimensions
            ax.set_xlim([0 - 1/self.plot_scale, (self.width + 1/self.plot_scale) * self.plot_scale + 1 + 1])
            ax.set_xticks(np.arange(0 - 1/self.plot_scale, self.width + 2/self.plot_scale, 1 / self.plot_scale))
            ax.set_ylim([0 - 1/self.plot_scale, (self.height + 1/self.plot_scale) * self.plot_scale + 1])
            ax.set_yticks(np.arange(0 - 1/self.plot_scale, self.height + 2/self.plot_scale, 1 / self.plot_scale))
            
            return ax
    
        elif plot_type == 'dash':

            fig = go.Figure()
            
            #figure_objects = []

            # add outline of map
            # Set axes properties
            fig.update_xaxes(range=[0, self.width], showgrid=False)
            fig.update_yaxes(range=[0, self.height], showgrid=False)
            # patches.Rectangle((0, 0), self.width / 2, self.height / 2, linewidth=1, edgecolor='b', facecolor='none')
            
            # add robot
            robo_mid = [self.robo.X, self.robo.Y] #[self.width / 2, self.height / 2]
            robo_shape = [ [ robo_mid[0] - 0.5, robo_mid[0], robo_mid[0] + 0.5, robo_mid[0] - 0.5 ], [ robo_mid[1] - 0.5, robo_mid[1] + 0.5, robo_mid[1] - 0.5, robo_mid[1] - 0.5 ] ]
            robo = go.Scatter(
                x = robo_shape[0],
                y = robo_shape[1],
                fill="toself" 
            )

            fig.add_trace(robo)

            # Add objects to map
            all_segments = []
            for obj in self.objects:
                all_segments.append(obj.get_points())
                fig.add_shape(type="rect",
                    x0=obj.pos[0], y0=obj.pos[1], 
                    x1=obj.pos[0] + obj.shape[0], y1=obj.pos[1] + obj.shape[1],
                    line=dict(
                        color="RoyalBlue",
                        width=2,
                        ),
                    fillcolor="LightSkyBlue",
                    )
                
            all_segments = np.array(all_segments)
            
            all_segments = all_segments.reshape(all_segments.shape[0] * all_segments.shape[1], all_segments.shape[2], all_segments.shape[3])

            # go.Figure([
            #     go.Scatter(
            #     x=[0,0,1,0],
            #     y=[0,2,1,0],
            #     fill="toself"),

            #     go.Scatter(
            #     x=[0, 1, 2, 0], 
            #     y=[0, 1, 0, 0], # replace with your own data source
            #     fill="toself"
            #     )
            # ])


            # fig.add_shape(type="rect",
            #     x0=1, y0=1, x1=2, y1=3,
            #     line=dict(color="RoyalBlue"),
            #     )

            # show lidar scans
            if show_lidar:

                total_scans = self.robo.ranges.shape[0]
                
                
                for laser in self.robo.get_lidar_ranges():
                    
                    startpos, endpos = laser
                    
                    all_intersecs = []
                    
                    ray_intersec = None
                    min_d = np.linalg.norm(endpos - startpos)
                    
                    for seg in all_segments:
                        
                        intersec = self.get_intersect(startpos, endpos, seg[0], seg[1])
                        if intersec is not None:
                            # get L2 length of vector
                            d = np.linalg.norm(intersec - startpos)
                            if d < min_d:
                                min_d = d
                                ray_intersec = intersec
                                
                    if ray_intersec is not None:
                        # fig.add_shape()

                        scatter_plot = self._draw_scatter(ray_intersec, plot_type=plot_type)

                        fig.add_trace(scatter_plot)
                        
                        # self._draw_scatter(ray_intersec, plot_type=plot_type) 

                    #self._draw_line(ax, startpos, endpos, ls='--')
                    # ax.plot([startp[0], endp[0]], [startp[1], endp[1]], c=c, ls=ls)

                    fig.add_shape(type="line",
                            xref="x", yref="y",
                            x0=startpos[0], y0=startpos[1], 
                            x1=endpos[0], y1=endpos[1],
                            line=dict(
                            color="DarkOrange",
                            width=3,
                            dash='dot'
                            )
                        )


            return fig
    

class Object():
    """Object class for Objects in a 2D space"""
    def __init__(self, pos=(0,0), shape=(0,0), random=True): #X=0, Y=0, width=0, height=0
        if random:
            pos, shape = self.generate_randomly() #X, Y, width, height
        self.pos = np.array(pos)
        self.shape = np.array(shape)
        
        #self.X = X
        #self.Y = Y
        #self.width = width
        #self.height = height
    
    
    def generate_randomly(self, map_wid = 20, map_hei = 20, min_wid = 1, min_hei = 1):
        X = int(np.random.rand() * (map_wid - min_wid))
        Y = int(np.random.rand() * (map_hei - min_hei))
        width = np.random.randint(min_wid, map_wid - X ) #* 0.01 # scale width in cm
        height = np.random.randint(min_hei, map_hei - Y) #* 0.01 # scale height in cm
        return (X, Y), (width, height)
    
    def get_points(self):
        """
        returns segments for each side of object
        bottom, left, up, right
        """
        
        wid_shift = np.zeros(self.pos.shape[0])
        wid_shift[0] = self.shape[0]
        heig_shift = np.zeros(self.pos.shape[0])
        heig_shift[1] = self.shape[1]
        
        seg_bottom = [self.pos, self.pos + wid_shift]
        seg_left = [self.pos, self.pos + heig_shift]
        seg_right = [self.pos + wid_shift, self.pos + wid_shift + heig_shift]
        seg_top = [self.pos + heig_shift, self.pos + wid_shift + heig_shift]
        
        
        return [seg_bottom, seg_left, seg_right, seg_top]
    
    

class Robot():
    def __init__(
                self, 
                origin, 
                vel, 
                rot = np.pi, 
                trans_vel = 0.,
                rot_vel = 0.,
                time_step = 0.1,
                laser_range = 5, 
                total_lidar_scans = 10, 
                lidar_scan_std = 5,
                use_filter = True,
                use_error = True,
                state_trans_func = None,
                measure_func = None,
                filter_func = None
            ):
        self.X, self.Y = origin
        self.vel = vel # velocity in m/s
        self.angle = rot
        #self.rot = rot

        self.trans_vel = trans_vel 
        self.rot_vel = rot_vel 
        self.motion_model = sample_motion_model_velocity(time_step = time_step)

        self.time_step = time_step

        self.laser_range = laser_range
        
        self.use_filter = use_filter 
        self.use_error = use_error

        if self.use_filter:

            if state_trans_func is not None:
                self.gx = state_trans_func
            else:                        
                def trans_func(u, sigma_points):
                    sigma_points_prime = np.zeros(sigma_points.shape)
                    motion_model = sample_motion_model_velocity(time_step = time_step)
                    for idx, sigma_point in enumerate(sigma_points):
                        sigma_points_prime[idx] = motion_model.forward(u, sigma_point)
                    return sigma_points_prime
                self.gx = trans_func #lambda u, x: x + 1 #trans_func
            
            if measure_func is not None:
                self.hx = measure_func
            else: 
                self.hx = lambda x: x

            if filter_func is not None:
                self.filter = filter_func 
            else:
                self.filter = UKF2D(
                                mu = np.zeros(3),
                                sigma = np.eye(3), 
                                g = self.gx,
                                h = self.hx)

        self.belief = {'mu': np.zeros([2]), 'sigma': np.ones([2,2])}

        self.ranges = np.zeros([total_lidar_scans + np.random.randint(lidar_scan_std),2,2])
        self._calc_lidar_ranges()
        
    def get_angle(self, radian=False):
        if radian:
            return self.angle 
        else:
            return 360 * self.angle / ( 2* np.pi )

    def update_vel(self, trans_vel, rot_vel):
        self.trans_vel = trans_vel 
        self.rot_vel = rot_vel 

    def move_motion_model(self, time):
        u = np.array([self.trans_vel, self.rot_vel])
        x_prev = np.array([self.X, self.Y, self.angle])
        
        assert(time > self.time_step)
        for t in range(int(time//self.time_step)):
            x_prime, y_prime, theta_prime = self.motion_model.forward(u, x_prev)
        
        self.X, self.Y = np.array([x_prime, y_prime])
        self.angle = theta_prime

        self._calc_lidar_ranges()
        ranges = self.get_lidar_ranges()

        if self.use_filter:
            # Update belief for position

            z = np.array([self.X, self.Y, self.angle])

            mu, sigma = self.filter.UKF(u = u, z = z)
            self.belief['mu'] = mu 
            self.belief['sigma'] = sigma

        return self.X, self.Y, self.angle
        
    def move(self, time, uncertainty = 0.2):
        """ time in seconds """

        # Update actual position
        self.X += round(self.get_ank(self.angle, self.vel * time, uncertainty = uncertainty), 2)
        self.Y += round(self.get_gek(self.angle, self.vel * time, uncertainty = uncertainty), 2)

        # TODO: use odom sensors ??? 

        self._calc_lidar_ranges()
        ranges = self.get_lidar_ranges()

        if self.use_filter:
            # Update belief for position

            z = np.array([self.X, self.Y])

            mu, sigma = self.filter.UKF(z)
            self.belief['mu'] = mu 
            self.belief['sigma'] = sigma

        # for t in range(time):
        #     self.X += round(self.get_ank(self.angle, self.vel), 2)
        #     self.Y += round(self.get_gek(self.angle, self.vel), 2)

        
        return self.X, self.Y
    
    def rotate(self, angle):
        """ time in seconds """
        self.angle += angle
        self._calc_lidar_ranges()
        
    def get_ank(self, angle, hyp, uncertainty = 0.2):
        if self.use_error:
            return np.cos(angle + np.random.randn() * uncertainty ) * hyp + np.random.randn() * uncertainty
        return np.cos(angle) * hyp

    def get_gek(self, angle, hyp, uncertainty = 0.2):
        if self.use_error:
            return np.sin(angle + np.random.randn() * uncertainty ) * hyp + np.random.randn() * uncertainty
        return np.sin(angle) * hyp
    
    def _rotate_vector(self, theta, v):
        rot_matr = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        return rot_matr @ v
    
    def _calc_lidar_ranges(self, uncertainty = 0.01):
        #ranges = np.zeros([,2])
        for i in range(self.ranges.shape[0]):
            if self.use_error:
                ang = (i + (np.random.randn() * uncertainty) ) * ( (2 * np.pi) / len(self.ranges) )
            else:
                ang = i * ( (2 * np.pi) / len(self.ranges) )
            startp = np.array([self.X, self.Y])
            endp = startp + self._rotate_vector(ang, self.laser_range * np.array([1, 0]))
            vec = np.array([startp, endp])
            
            self.ranges[i] = vec
            
        #self.ranges = np.array([i * ( (2 * np.pi) / len(self.ranges) ) for i in range(len(self.ranges))])
        
    def get_lidar_ranges(self): #, robomap):
        return self.ranges
    
        #self.ranges = []
        #for i in range(365):
        #    1/i * (np.pi * 2) 
    
    def estimate_pos(self):
        
        # state transition function - predict next state based
        # on constant velocity model x = vt + x_0
        gx = self.state_trans_func #lambda x : x + 1

        # measurement function - convert state into a measurement
        # where measurements are [x_pos, y_pos]
        hx = lambda x : x

        self.mu, self.sigma = UKF2D(u = None, g = gx, h = hx)

    def update_belief(self, mu, sigma):
        self.belief = {'mu': mu, 'sigma': sigma}
        return self.belief
    
    def get_belief(self):
        return {'mu': np.round(self.belief['mu'], 2), 'sigma': np.round(self.belief['sigma'], 2)}

        
        

        
        