from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

# import numpy as np

from env import (RoboMap, Object, Robot)


class Config():
    def __init__(self, env=None):
        
        self.reinit_count = 0
        self.reinit = True

        self.move_count = 0
        self.move = False

        self.rot_count = 0
        self.rot = False

        if env is not None:
            self.set_env(env)

    def set_value(self, key, value):
        setattr(self, key, value)
        return {key: value}
    
    def get_value(self, key):
        return getattr(self, key)

    def set_env(self, env):
        self.map = env['map']
        self.objects = env['objects']
        self.robo = env['robo']

    def get_env(self):
        return {
        'map': self.map,
        'objects': self.objects,
        'robo': self.robo
        }
    
    def inc_reinit_count(self):
        self.reinit_count += 1
        return self.reinit_count

    def inc_move_count(self):
        self.move_count += 1
        return self.move_count

    def inc_rot_count(self):
        self.rot_count += 1
        return self.rot_count
    

def init_env():
    rbMap = RoboMap(20, 20)
    obj1 = Object()
    rbMap.add_object(obj1)
    obj2 = Object()
    rbMap.add_object(obj2)
    robo1 = Robot(origin=(5,5), vel=0.05)
    rbMap.add_robo(robo1)
    env = {
        'map': rbMap,
        'objects': [obj1, obj2],
        'robo': robo1
        }

    fig = rbMap.draw_map(show_lidar=True, plot_type='dash')
    
    return fig, env

fig, env = init_env()

config = Config(env)

app = Dash(__name__)


app.layout = html.Div([
    html.H4('Robot Lidar 2D simulation'),
    dcc.Graph(figure=fig, id="robo_sim", style={'width': '90vh', 'height': '90vh'}),
    html.Button("Reinitialize environment", n_clicks=0, 
                id='shapes_reload_btn'),
    html.Button("Move robot", n_clicks=0,
                id='robo_btn_mv'),
    html.Button("Rotate robot", n_clicks=0,
                id='robo_btn_rot'),
])

@app.callback(
    Output("robo_sim", "figure"),
    Input("shapes_reload_btn", "n_clicks"),
    Input("robo_btn_mv", "n_clicks"),
    Input("robo_btn_rot", "n_clicks"))
def run_robo_sim(reload_clicks, mv_clicks, rot_clicks):
    global config

    if reload_clicks != config.get_value('reinit_count'):

        print('***** reinitialize environment *****')

        config.inc_reinit_count()

        fig, env = init_env()
        config.set_env(env)

    if mv_clicks != config.get_value('move_count'):

        print('***** moving robot *****')
        
        config.inc_move_count()

        env = config.get_env()

        robo = env['robo']
        robo.move(2)

        print('updated robot position')
        print(f'robo.X {robo.X}')
        print(f'robo.Y {robo.Y}')

        rbMap = env['map']

        rbMap.update_robot(robo)

        fig = rbMap.draw_map(show_lidar=True, plot_type='dash')
    
    if rot_clicks != config.get_value('rot_count'):

        print('***** rotating robot *****')
        
        config.inc_rot_count()

        env = config.get_env()

        robo = env['robo']
        robo.rotate(45)

        print('updated robot position')
        print(f'robo.X {robo.X}')
        print(f'robo.Y {robo.Y}')

        rbMap = env['map']

        rbMap.update_robot(robo)

        fig = rbMap.draw_map(show_lidar=True, plot_type='dash')

    if config.get_value('reinit'):
        fig, env = init_env()
        config.set_env(env)

        config.set_value('reinit', False)

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
