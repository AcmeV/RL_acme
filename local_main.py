import argparse

from utils.training.gym_train import gym_train
from utils.training.pygame_train import pygame_train
from utils.training.tkinter_train import tkinter_train

parser = argparse.ArgumentParser()
# System settings
parser.add_argument('--data-dir', type=str, default='./data/')
parser.add_argument('--config-dir', type=str, default='./config/')
parser.add_argument('--model-save-dir', type=str, default='./model_pkl/')
parser.add_argument('--log-dir', type=str, default='./logs')
parser.add_argument('--device', type=str, default='cpu',
                    choices=('cpu', 'gpu', 'gpus'))
parser.add_argument('--gpus', type=str, default='0,1,2,3')

# env
parser.add_argument('--env-type', type=str, default='tkinter',
                    choices=('tkinter', 'gym', 'pygame'))

parser.add_argument('--env-name', type=str, default='Maze',
                    choices=('Maze', 'Maze_v0',
                             'FlappyBird', 'Snake'
                             'Pendulum', 'MountainCar', 'CartPole'))
parser.add_argument('--is-render', type=int, default=1, choices=(0, 1))

# Hyper-parameters
parser.add_argument('--model', type=str, default='QLearning',
                    choices=('QLearning', 'Sarsa', 'SarsaLambda',
                             'DQN', 'DoubleDQN'))
parser.add_argument('--episodes', type=int, default=150)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--pre-training', type=int, default=1, choices=(0, 1))
parser.add_argument('--if-save', type=int, default=1, choices=(0, 1))
args = parser.parse_args()

if __name__ == '__main__':

    args.env_type = 'tkinter'
    args.env_name = 'Maze_v0'
    args.model = 'DQN'

    args.pre_training = 1
    args.if_save = 0

    args.is_render = 0

    args.episodes = 5000
    args.lr = 0.1

    if args.env_type == 'tkinter':
        tkinter_train(args)
    elif args.env_type == 'pygame':
        pygame_train(args)
    else:
        gym_train(args)
