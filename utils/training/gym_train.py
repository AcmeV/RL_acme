import csv
import os

import numpy as np
import torch

from model import QLearningTable
from model.DQN import DQN
from utils import initial_env
from model.DQN import DoubleDQN

def qlearning_training(env, q_table, episodes, save_path, log_path):

    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Loss', 'Info', 'Step'])

    """主程序"""
    observation = env.reset()

    total_steps = 0

    for episode in range(episodes):

        if env.has_terminal_tag:
            observation = env.reset()

        terminate = False

        step_counter = 0

        episode_losses = []
        episode_rewards = []

        while not terminate:

            env.render()

            action = q_table.choose_action(observation)

            observation_, reward, terminate, info = env.step(action)

            loss = q_table.learn(observation, action, reward, observation_)

            episode_losses.append(loss)
            episode_rewards.append(reward)

            observation = observation_
            step_counter += 1
            total_steps += 1

            if not env.has_terminal_tag and total_steps % 2000 == 0:
                terminate = True

        episode_loss = np.mean(episode_losses)
        episode_reward = np.mean(episode_rewards)

        writer.writerow([episode, episode_loss, episode_reward, step_counter])
        file.flush()

        print(f'Episode: {episode} | Loss: {episode_loss: .4f} | Reward: {episode_reward: .4f} | Step: {step_counter}')

    if save_path is not None:
        q_table.save(episodes, save_path)

    file.close()

def dqn_training(env, model, episodes, save_path, log_path):

    file = open(log_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Loss', 'Info', 'Step'])

    total_steps = 1
    memory_size = model.memory_size

    observation = env.reset()

    for episode in range(episodes):

        if env.has_terminal_tag:
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

            if total_steps > memory_size:
                episode_losses.append(model.learn())
            episode_rewards.append(reward + 1)

            if not env.has_terminal_tag and total_steps > memory_size and total_steps % memory_size == 0:
                terminate = True

            observation = observation_
            total_steps += 1
            step_counter += 1

            if terminate and total_steps <= memory_size:
                step_counter = 0
                episode_losses = []
                episode_rewards = []
                terminate = False
                if env.has_terminal_tag:
                    observation = env.reset()


        episode_loss = np.mean(episode_losses)
        episode_reward = np.mean(episode_rewards)

        writer.writerow([episode, episode_loss, episode_reward, step_counter])
        file.flush()
        print(f'Episode: {episode} | Loss: {episode_loss: .4f}| Reward: {episode_reward: .4f} | Step: {step_counter}')

    if save_path is not None:
        model.save(episodes, save_path)

    file.close()

def gym_train(args):

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

        qlearning_training(env, q_table, args.episodes, save_path, log_path)
    elif 'DQN' in args.model:
        device = torch.device(f'cuda:0' if args.device != 'cpu'
                                               and torch.cuda.is_available() else "cpu")
        if args.model == 'DQN':
            model = DQN(env.n_actions, env.n_features,
                        learning_rate=args.lr, reward_decay=0.9,
                        e_greedy=0.9, e_greedy_increment=0.00005,
                        replace_target_iter=200, memory_size=2000, device=device)
        elif args.model == 'DoubleDQN':

            model = DoubleDQN(env.n_actions, env.n_features,
                        learning_rate=args.lr, reward_decay=0.9,
                              e_greedy=0.9, e_greedy_increment=0.00005,
                        replace_target_iter=100, memory_size=2000, device=device)
        else:
            model = DQN(env.n_actions, env.n_features,
                        learning_rate=args.lr, reward_decay=0.9,
                        e_greedy=0.9, e_greedy_increment=0.00005,
                        replace_target_iter=200, memory_size=2000, device=device)
        # load pre-training model parameters
        if os.path.exists(load_path) and args.pre_training == 1:
            model.load(load_path)
        dqn_training(env, model, args.episodes, save_path, log_path)

    print('training over')
