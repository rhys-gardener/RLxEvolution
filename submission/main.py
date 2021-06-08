from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras import backend as K
import tensorflow as tf

import kaggle_environments as kaggle

import numpy as np

class DiscreteActorModelKaggle():
    def __init__(self, input_shape, action_space):

        X_input = Input(input_shape)
        self.action_space = action_space

        X = Conv2D(160,1, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Conv2D(80,1, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Conv2D(64,1, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Flatten()(X)
        output = Dense(self.action_space, activation="softmax")(X)

        self.Actor = Model(inputs = X_input, outputs = output)

    def predict(self, state):
        return self.Actor.predict(state)


def get_geese_coord(board):
    return get_coord_from_np_grid(board, 101)

def get_food_coord(board):
    return get_coord_from_np_grid(board, 1000)

def get_enemy_geese_head_coord(board):
    return get_coord_from_np_grid(board, -99)


def get_coord_from_np_grid(grid, value):
    coords = []
    for i in range(0, len(np.where(grid==value)[0])):
        coords.append((np.where(grid==value)[0][i], np.where(grid==value)[1][i]))
    return coords

def get_geese_observation(rows, columns, agent, observation):
    """
    Given a particular geese, does some processing and returns a geese specific observation. 
    Unfortunately specific to the geese environment for now.
    Encoding as follows: 
    2: enemy snake head
    1: enemy snake body
    11: own head
    12: own body
    100: food
    """

    game_board_self = np.zeros(rows*columns, None)
    game_board_enemy = np.zeros(rows*columns, None)
    game_board_food = np.zeros(rows*columns, None)


    for i, geese in enumerate(observation.geese):
        identify=0
        if i==agent:
            identify=100
            for j, cell in enumerate(geese):
                if j == 0:
                    game_board_self[cell] = identify+1
                else:
                    game_board_self[cell] = identify+2
        else:
            identify=-100
            for j, cell in enumerate(geese):
                if j == 0:
                    game_board_enemy[cell] = identify+1
                else:
                    game_board_enemy[cell] = identify+2
            
    for food in observation.food:
        game_board_food[food] = 1000
    game_board_self = game_board_self.reshape([rows, columns])
    game_board_enemy = game_board_enemy.reshape([rows, columns])
    game_board_food = game_board_food.reshape([rows, columns])

    head = get_geese_coord(game_board_self)

    if len(head)==0:
        head = (0,0) # doesn't matter in submission
    else:
        head = head[0]
        game_board_self = np.roll(game_board_self, 5-head[1], axis=1)
        game_board_self = np.roll(game_board_self, 3-head[0], axis=0)
        game_board_enemy = np.roll(game_board_enemy, 5-head[1], axis=1)
        game_board_enemy = np.roll(game_board_enemy, 3-head[0], axis=0)
        game_board_food = np.roll(game_board_food, 5-head[1], axis=1)
        game_board_food = np.roll(game_board_food, 3-head[0], axis=0)

    #game_board = game_board.reshape((game_board.shape[0], game_board.shape[1], 1))
    game_board = np.dstack((game_board_self, game_board_enemy, game_board_food))
    return game_board


def agent(obs_dict, config_dict):
    #load weights
    actions = ['NORTH','SOUTH','WEST','EAST']
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    rows = configuration.rows
    columns = configuration.columns
    observation_space = (rows, columns, 3)

    player_index = observation.index
    Actor = DiscreteActorModelKaggle(observation_space, 4)
    Actor.Actor.load_weights('/kaggle_simulations/agent/champion_actor_6.h5')

    state = get_geese_observation(rows, columns, player_index, observation)
    state = np.expand_dims(state, axis=0)
    prediction = Actor.predict(state)[-1]
    action = actions[np.argmax(prediction)]
    return action
