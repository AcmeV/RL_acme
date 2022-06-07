import csv
import os

import numpy as np
import torch

from model.DQN import DQN, DoubleDQN
from utils import initial_env
from model import SarsaLambdaTable
from model.epsilon_greedy.SarsaTable import SarsaTable
from model.epsilon_greedy.QLearningTable import QLearningTable


def qlearning_training(env, q_table, episodes, save_path, log_path):

    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Loss', 'Info', 'Step'])

    for episode in range(episodes):

        observation = env.reset()

        terminate = False

        step_counter = 0
        episode_losses = []
        episode_rewards = []

        while not terminate:

            env.render()

            action = q_table.choose_action(observation)

            observation_, reward, terminate, info = env.step(action)

            episode_losses.append(q_table.learn(observation, action, reward, observation_))
            episode_rewards.append(reward)
            step_counter += 1

            observation = observation_

        episode_loss = np.mean(episode_losses)
        episode_reward = np.mean(episode_rewards)
        writer.writerow([episode, episode_loss, episode_reward, step_counter])
        print(f'Episode: {episode} | Loss: {episode_loss: .4f}| Reward: {episode_reward: .4f} | Step: {step_counter}')

    if save_path is not None:
        q_table.save(episodes, save_path)
    file.close()
    env.destroy()

def sarsa_training(env, sarsa_table, episodes, save_path, log_path):

    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Loss', 'Info', 'Step'])

    for episode in range(episodes):

        observation = env.reset()

        terminate = False

        step_counter = 0
        episode_losses = []
        episode_rewards = []

        action = sarsa_table.choose_action(observation)

        while not terminate:

            env.render()
            observation_, reward, terminate, info = env.step(action)

            action_ = sarsa_table.choose_action(observation)

            episode_losses.append(sarsa_table.learn(observation, action, reward, observation_, action_))
            episode_rewards.append(reward)
            step_counter += 1

            observation = observation_
            action = action_

        episode_loss = np.mean(episode_losses)
        episode_reward = np.mean(episode_rewards)
        writer.writerow([episode, episode_loss, episode_reward, step_counter])
        print(f'Episode: {episode} | Loss: {episode_loss: .4f}| Reward: {episode_reward: .4f} | Step: {step_counter}')

    if save_path is not None:
        sarsa_table.save(episodes, save_path)
    file.close()
    env.destroy()

def dqn_training(env, model, episodes, save_path, log_path):

    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Loss', 'Info', 'Step'])

    total_counter = 0

    for episode in range(episodes):

        observation = env.reset()

        terminate = False

        step_counter = 0
        episode_losses = []
        episode_rewards = []

        while not terminate:

            env.render()

            action = model.choose_action(observation)

            observation_, reward, terminate, info = env.step(action)

            model.store_transition(observation, action, reward, observation_)

            if total_counter > 32:
                episode_losses.append(model.learn())

            episode_rewards.append(reward)
            step_counter += 1
            total_counter += 1
            observation = observation_

            if terminate and total_counter <= 32:
                terminate = False
                step_counter = 0
                episode_losses = []
                episode_rewards = []
                observation = env.reset()

        episode_loss = np.mean(episode_losses)
        episode_reward = np.mean(episode_rewards)
        writer.writerow([episode, episode_loss, episode_reward, step_counter])
        print(f'Episode: {episode} | Loss: {episode_loss: .4f}| Reward: {episode_reward: .4f} | Step: {step_counter}')

    if save_path is not None:
        model.save(episodes, save_path)
    env.destroy()

def tkinter_train(args):

    env = initial_env(args)
    if args.is_render == 1:
        env.is_render = True

    save_path = f'{args.model_save_dir}/{args.model}-{args.env_type}-{args.env_name}'
    load_path = f'{args.model_save_dir}/{args.model}-{args.env_type}-{args.env_name}'

    log_path = f'{args.log_dir}/{args.model}-{args.env_type}-{args.env_name}.csv'

    if args.if_save == 0:
        save_path = None

    if args.model == 'QLearning':

        q_table = QLearningTable(
            actions=list(range(env.n_actions)),
            learning_rate=args.lr)
        # load pre-training model parameters
        if os.path.exists(load_path) and args.pre_training == 1:
            q_table.load(load_path)

        env.after(100, qlearning_training, env, q_table,
                  args.episodes, save_path, log_path)
    elif args.model == 'Sarsa':

        sarsa_table = SarsaTable(
            actions=list(range(env.n_actions)),
            learning_rate=args.lr)
        # load pre-training model parameters
        if os.path.exists(load_path) and args.pre_training == 1:
            sarsa_table.load(load_path)

        env.after(100, sarsa_training, env, sarsa_table,
                  args.episodes, save_path, log_path)
    elif args.model == 'SarsaLambda':
        sarsa_table = SarsaLambdaTable(
            actions=list(range(env.n_actions)),
            learning_rate=args.lr)
        # load pre-training model parameters
        if os.path.exists(load_path) and args.pre_training == 1:
            sarsa_table.load(load_path)

        env.after(100, sarsa_training, env, sarsa_table,
                  args.episodes, save_path, log_path)
    elif 'DQN' in args.model:
        device = torch.device(f'cuda:0' if args.device != 'cpu'
                                           and torch.cuda.is_available() else "cpu")
        if args.model == 'DQN':
            model = DQN(env.n_actions, env.n_features,
                        learning_rate=args.lr,
                        reward_decay=0.9, e_greedy=0.9,
                        replace_target_iter=200, memory_size=2000, device=device)
        elif args.model == 'DoubleDQN':
            model = DoubleDQN(env.n_actions, env.n_features,
                        learning_rate=args.lr,
                        reward_decay=0.9, e_greedy=0.9,
                        replace_target_iter=200, memory_size=2000, device=device)
        else:
            model = DQN(env.n_actions, env.n_features,
                        learning_rate=args.lr,
                        reward_decay=0.9, e_greedy=0.9,
                        replace_target_iter=200, memory_size=2000, device=device)
        # load pre-training model parameters
        if os.path.exists(load_path) and args.pre_training == 1:
            model.load(load_path)

        env.after(100, dqn_training, env, model,
                  args.episodes, save_path, log_path)


    env.mainloop()

    print('training over')
