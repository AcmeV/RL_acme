import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean


font = {
    'size': 13
}

def load_log(path):
    log = pd.read_csv(path)
    episodes, steps, losses = [], [], []

    # for i in range(len(log['Epoch'])):
    for i in range(len(log['Episode'])):
        episodes.append(log['Episode'][i])
        steps.append(float(log['Info'][i]))
        losses.append(float(log['Loss'][i]))

    return episodes, steps, losses

def step_plot(episodes, steps, colors, labels, title):
    plt.figure(1, figsize=(6, 5))

    plt.title(title)
    plt.xlabel('Episode', font)
    plt.ylabel('Step', font)

    for i in range(len(episodes)):
        plt.plot(episodes[i], steps[i], color=colors[i], label=labels[i])
    plt.legend(loc='upper right', fontsize=12, ncol=2)
    plt.show()

def loss_plot(episodes, losses, colors, labels, title):
    plt.figure(1, figsize=(6, 5))

    plt.title(title)
    plt.xlabel('Episode', font)
    plt.ylabel('Loss', font)
    for i in range(len(episodes)):
        plt.plot(episodes[i], losses[i], color=colors[i], label=labels[i])

    plt.legend(loc='upper right', fontsize=12, ncol=2)
    plt.show()

def plot_diff_model():

    colors = ['green', 'steelblue', 'red']
    labels = [
        'QLearning',
        'Sarsa',
        'Sarsa(λ)'
    ]
    models = [
        'QLearning',
        'Sarsa',
        'SarsaLambda'
    ]
    paths = [f'./diff_model/{model}-tkinter-Maze_v0.csv' for model in models]
    step_title = 'Step comparision of diffrent model\nLR: 0.1 | Env: Maze'
    loss_title = 'Loss comparision of diffrent model\nLR: 0.1 | Env: Maze'

    episodes, steps, losses = [], [], []
    for i in range(len(labels)):
        episode, step, loss = load_log(paths[i])
        episodes.append(episode)
        steps.append(step)
        losses.append(loss)

    step_plot(episodes, steps, colors, labels, step_title)
    loss_plot(episodes, losses, colors, labels, loss_title)

def plot_diff_reward():

    colors = ['green', 'steelblue', 'red']
    labels = [
        'reward = 1',
        'reward = 10',
        'reward = 100'
    ]
    rewards = [
        '1',
        '10',
        '100'
    ]
    paths = [f'./diff_reward/reward_{reward}.csv' for reward in rewards]
    step_title = 'Step comparision of diffrent reward\nModel: Q-learning | LR: 0.1 | Env: Maze'
    loss_title = 'Loss comparision of diffrent reward\nModel: Q-learning | LR: 0.1 | Env: Maze'

    episodes, steps, losses = [], [], []
    for i in range(len(labels)):
        episode, step, loss = load_log(paths[i])
        episodes.append(episode)
        steps.append(step)
        losses.append(loss)

    step_plot(episodes, steps, colors, labels, step_title)
    loss_plot(episodes, losses, colors, labels, loss_title)

def plot_diff_gamma():

    colors = ['green', 'steelblue', 'orange', 'red']
    labels = [
        'gamma = 0',
        'gamma = 0.5',
        'gamma = 0.9',
        'gamma = 1',
    ]
    gammas = [
        '0',
        '5',
        '9',
        '10',
    ]
    paths = [f'./diff_gamma/gamma_{gamma}.csv' for gamma in gammas]
    step_title = 'Step comparision of diffrent γ\nModel: Q-learning | LR: 0.1 | Env: Maze'
    loss_title = 'Loss comparision of diffrent γ\nModel: Q-learning | LR: 0.1 | Env: Maze'

    episodes, steps, losses = [], [], []
    for i in range(len(labels)):
        episode, step, loss = load_log(paths[i])
        episodes.append(episode)
        steps.append(step)
        losses.append(loss)

    step_plot(episodes, steps, colors, labels, step_title)
    loss_plot(episodes, losses, colors, labels, loss_title)

if __name__ == '__main__':
    # plot_diff_model()
    # plot_diff_gamma()
    plot_diff_reward()