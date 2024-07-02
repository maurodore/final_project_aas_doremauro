import warnings
import numpy as np
from utilities import train_game, test_game, test_random

from parameters import SEED

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set the random seed for reproducibility
np.random.seed(SEED)

game_name = 'coinrun'
learning_rate = 5e-4

#Function to train the agent in a specified game and learning rate
train_game(game_name=game_name, learning_rate=learning_rate)

#Function to test the agent in a specified game
#test_game(game_name=game_name, learning_rate=learning_rate, file_name='weights_tested/coinrun_0.0001_200levels.weights.h5')

#Function to test the random agent in a specified game
#test_random(game_name=game_name)
