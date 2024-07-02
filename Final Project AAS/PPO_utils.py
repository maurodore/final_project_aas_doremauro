import numpy as np
from network import ImpalaModelV2
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow_probability as tfp

from parameters import (SEED, GAMMA, GAE_LAMBDA, CLIP_PARAMETER, N_EPOCHS, ENTROPY_COEFF,
                        MINIBATCH_SIZE, N_LEVELS)

# Set the random seed for reproducibility
tf.random.set_seed(SEED)


def calculate_gae(rewards, values, done, gamma=0.99, lambda_gae=0.99):
    """
        Calculate the Generalized Advantage Estimation (GAE) for policy updates.

        Args:
            rewards (np.array): Rewards obtained from the environment.
            values (np.array): Value estimates as calculated by the critic.
            done (np.array): Boolean flags indicating the end of an episode.
            gamma (float): Discount factor for future rewards.
            lambda_gae (float): Smoothing factor to balance bias vs variance.

        Returns:
            np.array: Computed advantages.
    """

    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_advantage = 0.0

    last_value = values[-1]

    for t in reversed(range(len(rewards))):
        mask = 1.0 - done[t]
        last_value = last_value * mask
        last_advantage = last_advantage * mask
        delta = rewards[t] + gamma * last_value - values[t]
        last_advantage = delta + gamma * lambda_gae * last_advantage
        advantages[t] = last_advantage
        last_value = values[t]

    return advantages


class PPOMemory:
    """
        Memory buffer for storing experience tuples and generating training batches for the PPO Agent.
    """
    def __init__(self):
        self.states = []
        self.probabilities = []
        self.value_functions = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = MINIBATCH_SIZE

    def generate_batches(self):
        """
            Create batches from stored experiences, shuffled for training diversity.

            Returns:
                tuple: Arrays of states, actions, old probabilities, value functions, rewards, dones, and indices for minibatches.
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:(i + self.batch_size)] for i in batch_start]

        return (np.array(self.states), np.array(self.actions), np.array(self.probabilities),
                np.array(self.value_functions), np.array(self.rewards), np.array(self.dones), batches)

    def store_memory(self, state, action, probabilities, value_function, reward, done):
        """
            Store experiences in the memory buffer.

            Args:
                state: The current state.
                action: The action taken.
                probabilities: The action probabilities.
                value_function: The value function output.
                reward: The received reward.
                done: Boolean flag for episode termination.
        """
        self.states.append(state)
        self.actions.append(action)
        self.probabilities.append(probabilities)
        self.value_functions.append(value_function)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """Clear all stored experiences in the memory."""
        self.states = []
        self.probabilities = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.value_functions = []


class PPOAgent:
    """
        Proximal Policy Optimization (PPO) Agent implementing the actor-critic method.
    """
    def __init__(self, n_actions, learning_rate):

        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.gae_lambda = GAE_LAMBDA
        self.clip_parameter = CLIP_PARAMETER
        self.n_epochs = N_EPOCHS
        self.entropy_coefficient = ENTROPY_COEFF

        self.actor_critic = ImpalaModelV2(n_actions)
        self.actor_critic_optimizer = Adam(learning_rate=self.learning_rate)

        #self.actor_critic.build((1, 64, 64, 3))
        #self.actor_critic.summary()

        self.memory = PPOMemory()

    def store_step(self, state, action, probabilities, value_function, reward, done):
        """
            Store the step taken in the environment to memory.

            Args:
                state: The current state of the environment.
                action: The action taken.
                probabilities: The output probabilities of the action.
                value_function: The estimated value of the state.
                reward: The received reward.
                done: Boolean flag indicating if the episode has ended.
        """
        self.memory.store_memory(state, action, probabilities, value_function, reward, done)

    def save_models(self, game_name, learning_rate):
        """
            Save the model weights.

            Args:
                game_name (str): Name of the game being played.
                learning_rate (float): Learning rate of the optimizer.
        """
        self.actor_critic.save_weights('weights/'+game_name+'_'+str(learning_rate)+'_'+str(N_LEVELS)+'levels.weights.h5')

    def load_models(self, file_name):
        """Load the model weights from a given file."""
        self.actor_critic.load_weights(file_name)

    def act(self, state):
        """
            Choose an action based on the current state.

            Args:
                state: The current state.

            Returns:
                tuple: Action, log probability of the action, and the state value.
        """
        state = np.expand_dims(state, axis=0)
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        probabilities, critic_value = self.actor_critic(state)

        probabilities_distribution = tfp.distributions.Categorical(probabilities)

        action = probabilities_distribution.sample()

        log_prob = tf.gather(probabilities, action, batch_dims=1)

        return action.numpy()[0], log_prob.numpy()[0], critic_value.numpy()[0][0]

    def learn(self):
        """
             Perform learning iterations for the model using the experiences stored in memory.
        """
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, values_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            advantage = calculate_gae(reward_arr, values_arr, dones_arr, gamma=self.gamma, lambda_gae=self.gae_lambda)

            for batch in batches:
                with tf.GradientTape() as tape:
                    states = tf.convert_to_tensor(state_arr[batch], dtype=tf.float32)
                    old_probabilities = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    probs, critic_value = self.actor_critic(states)

                    new_probabilities = tf.gather(probs, actions, batch_dims=1)

                    entropy = tf.reduce_mean(-tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))

                    policy_prob_ratio = tf.math.exp(new_probabilities - old_probabilities)

                    weighted_probs = policy_prob_ratio * advantage[batch]

                    clipped_probs = tf.clip_by_value(policy_prob_ratio, 1 - self.clip_parameter,
                                                     1 + self.clip_parameter)

                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)

                    actor_loss = tf.math.reduce_mean(actor_loss)

                    # - - - - - - - - - - - - - - - - - - - - - - - - #

                    critic_value = tf.squeeze(critic_value, 1)

                    returns = advantage[batch] + values_arr[batch]

                    critic_loss = tf.keras.losses.MSE(critic_value, returns)

                    actor_critic_loss = (actor_loss + (0.5 * critic_loss)) - self.entropy_coefficient * entropy

                actor_critic_grads = tape.gradient(actor_critic_loss, self.actor_critic.trainable_variables)
                self.actor_critic_optimizer.apply_gradients(
                    zip(actor_critic_grads, self.actor_critic.trainable_variables))

        self.memory.clear_memory()
