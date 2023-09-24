import argparse

import matplotlib.pyplot as plt

from env import (RoboMap, Object, Robot)


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

    return env

def simulate(
        reps,
        plot_graph = False,
        plot_name = 'filter_plot'
        ):
    """
        reps: no of repetitions
        plot_graph: should data be plotted in a graph
    """
    
    env = init_env()

    if plot_graph:
        fig = plt.figure()

    for i in range(reps):

        robo = env['robo']
        
        robo.update_vel(trans_vel=1 , rot_vel=0.5 )
        robo.move_motion_model(time = 1)
        #robo.move(2)

        print('------ ****** ------')

        print('updated robot position')
        print(f'robo.X {robo.X}')
        print(f'robo.Y {robo.Y}')
        print(f'robo.angle {robo.angle}')

        print('---')

        print('belief of robot position')
        print(f'robo belief expectation \n{robo.get_belief()["mu"]}')
        print(f'robo belief covariance \n{robo.get_belief()["sigma"]}')
        # print(f'robo Y belief expectation {robo.get_belief()["mu"][1]}')
        # print(f'robo Y belief covariance {robo.get_belief()["sigma"][1]}')

        print('------ ****** ------')

        rbMap = env['map']

        rbMap.update_robot(robo)

    if plot_graph:
        fig.savefig('')

if __name__ == '__main__':

    # DEFAULT SETTINGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=50, help='no of repetitions for simulation [default: 50]')
    parser.add_argument('--plot_graph', default='False', help='should a graph be plotted [default: False]')
    parser.add_argument('--plot_name', default='filter_plot', help='name for plotted graph; only relevant if plot_graph is True [default: filter_plot]')
    FLAGS = parser.parse_args()

    if FLAGS.plot_graph == 'True':
        plot_graph = True
    else:
        plot_graph = False

    simulate(FLAGS.reps, plot_graph, FLAGS.plot_name)