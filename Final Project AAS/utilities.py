import os
import time
import matplotlib.pyplot as plt
import numpy as np
import gym
from PPO_utils import PPOAgent

from parameters import (MAX_TIMESTEPS, USE_BACKGROUND, SEED, N_LEVELS, N_LEVELS_TEST, RENDER_MODE, N_FRAME_SKIP,
                        BATCH_SIZE)


def train_game(game_name, learning_rate):
    """
        Train the PPO agent on the specified game.

        Args:
            game_name (str): Name of the game to train on.
            learning_rate (float): Learning rate for the PPO agent.
    """
    env = gym.make('procgen:procgen-' + game_name + '-v0', num_levels=N_LEVELS, start_level=SEED,
                   rand_seed=SEED, distribution_mode='easy', use_backgrounds=False, render_mode=RENDER_MODE)

    action_size = env.action_space.n

    agent = PPOAgent(n_actions=action_size, learning_rate=learning_rate)

    n_episode = 0
    score_history = []
    timesteps_history = []
    avg_score_episode = []

    start_time = time.time()
    print_start_training(game_name, learning_rate)

    while sum(timesteps_history) <= MAX_TIMESTEPS:
        state = env.reset()
        done, score, n_steps = False, 0, 0

        while not done:
            action, prob, val = agent.act(state)
            for fs in range(N_FRAME_SKIP):
                next_state, reward, done, _ = env.step(action)

                normalised_reward = reward

                n_steps += 1
                score += reward

                if fs == 0:
                    agent.store_step(state, action, prob, val, normalised_reward, done)

                if len(agent.memory.states) >= BATCH_SIZE:
                    agent.learn()

                state = next_state

                if done:
                    timesteps_history.append(n_steps)
                    score_history.append(score)
                    avg_score_episode.append(score / n_steps)

                    break

        n_episode += 1

    print_end_training(score_history, timesteps_history, n_episode)

    agent.save_models(game_name=game_name, learning_rate=learning_rate)

    env.close()

    end_time = time.time()
    print("- " * 35 + "\n" + f"Execution time of Training: {(end_time - start_time) / 60:.1f} minutes")

    plot_moving_average(score_history, game_name=game_name, learning_rate=learning_rate, timesteps=timesteps_history)


def test_game(game_name, learning_rate, file_name, test_size=20):
    """
        Test the trained agent on the specified game.

        Args:
            game_name (str): Name of the game to test on.
            learning_rate (float): Learning rate used during training.
            file_name (str): Path to the model weights file.
            test_size (int): Number of episodes to test.
    """
    print("* " * 40 + "\n" + "PPO Test")

    env = gym.make('procgen:procgen-' + game_name + '-v0', num_levels=N_LEVELS_TEST, start_level=SEED,
                   rand_seed=SEED, distribution_mode='easy', use_backgrounds=USE_BACKGROUND, render_mode=RENDER_MODE)

    action_size = env.action_space.n
    agent = PPOAgent(n_actions=action_size, learning_rate=learning_rate)

    agent.actor_critic.build((1, 64, 64, 3))
    agent.load_models(file_name=file_name)
    score_history = []
    time_history = []
    n_episodes = 0
    max_timesteps = MAX_TIMESTEPS // test_size

    while sum(time_history) <= max_timesteps:
        state = env.reset()
        done, score, n_steps = False, 0, 0
        while not done:
            action, _, _ = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            n_steps += 1

        n_episodes += 1
        time_history.append(n_steps)
        score_history.append(score)

    print(f"Total avg score: {np.mean(score_history):.2f}, Number of episodes: {n_episodes}")
    print(f"Avg episodes lenght: {np.mean(time_history):.2f}")


def test_random(game_name, test_size=20):
    """
        Test the environment with random actions.

        Args:
            game_name (str): Name of the game to test.
            test_size (int): Number of episodes to test.
    """
    print("* " * 40 + "\n" + "Random Test")

    env = gym.make('procgen:procgen-' + game_name + '-v0', num_levels=N_LEVELS_TEST, start_level=SEED,
                   rand_seed=SEED, distribution_mode='easy', use_backgrounds=False, render_mode=RENDER_MODE)

    score_history = []
    time_history = []
    n_episodes = 0
    max_timesteps = MAX_TIMESTEPS // test_size
    print("Max Timesteps: ", max_timesteps)

    while sum(time_history) <= max_timesteps:
        state = env.reset()
        done, score, n_steps = False, 0, 0

        while not done:
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            score += reward
            n_steps += 1

        n_episodes += 1
        time_history.append(n_steps)
        score_history.append(score)

    print(f"Total avg score: {np.mean(score_history):.2f}, Number of episodes: {n_episodes}")
    print(f"Avg episodes lenght: {np.mean(time_history):.2f}")


def print_start_training(game_name, learning_rate):
    """
        Print the start message for training.

        Args:
            game_name (str): Name of the game being trained on.
            learning_rate (float): Learning rate used for training.
    """
    print("* " * 50 + "\n" + "PPO Training")

    print('Game: ' + game_name + ', Learning Rate: ' + str(learning_rate) + ', Number of timesteps: ' + str(
        MAX_TIMESTEPS) + ', Number of levels: ' + str(N_LEVELS))


def print_end_training(score_history, timesteps_history, n_episodes):
    """
        Print the end message with training results.

        Args:
            score_history (list): List of scores obtained for each episode during training.
            timesteps_history (list): List of timesteps for each episode taken during training.
            n_episodes (int): Number of episodes completed during training.
    """
    print("- " * 35 + "\n" + f"Avg first 100 score: {np.mean(score_history[:100]):.2f},"
                             f" Average first 100 timesteps: {np.mean(timesteps_history[:100]):.2f}")

    print("- " * 35 + "\n" + f"Avg last 100 score: {np.mean(score_history[-100:]):.2f},"
                             f" Average last 100 timesteps: {np.mean(timesteps_history[-100:]):.2f}")

    print("- " * 35 + "\n" + f"Total avg score: {np.mean(score_history):.2f}, Number of episoides: {n_episodes},"
                             f" Avg episodes lenght: {np.mean(timesteps_history):.2f}")


def plot_moving_average(rewards, game_name, learning_rate, timesteps, windows_size=50):
    """
        Plot and save the moving average of rewards over episodes.

        Args:
            rewards (list): List of rewards obtained during training.
            game_name (str): Name of the game being trained on.
            learning_rate (float): Learning rate used for training.
            timesteps (list): List of timesteps for each episode taken during training.
            windows_size (int): Window size for moving average.
    """
    filename = game_name + '_' + str(learning_rate) + '.png'
    figure_file = 'Plots/' + filename
    os.makedirs(os.path.dirname(figure_file), exist_ok=True)

    moving_average = np.convolve(rewards, np.ones(windows_size) / windows_size, mode='valid')

    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Scores')

    ax1.plot(moving_average, color='tab:blue', linewidth=2)
    ax1.plot(rewards[windows_size - 1:], color='tab:red', alpha=0.5)

    plt.suptitle('Average Reward Plot \nGame: ' + str(game_name) + ' ' +
                 '\nLearning Rate: ' + str(learning_rate) + ' Timesteps: ' + str(sum(timesteps)))

    plt.savefig(figure_file)
    plt.show()
    plt.close(fig)
