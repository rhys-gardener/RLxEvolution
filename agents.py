import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras import backend as K

import numpy as np

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action

GLOBAL_ACTIONS = ['NORTH','SOUTH','WEST','EAST']
from random import choice

class ContinuousActorModel():
    def __init__(self, input_shape, action_space, lr, optimizer):

        X_input = Input(input_shape)
        self.action_space = action_space
        """
        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        """
        X = Conv2D(160,1, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Conv2D(80,1, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Conv2D(64,1, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Flatten()(X)
        output = Dense(self.action_space, activation="tanh")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(lr=lr))

        self.log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        self.std = np.exp(self.log_std)

    def ppo_loss_continuous(self, y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]
        LOSS_CLIPPING = 0.2
        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING)*advantages, (1.0 - LOSS_CLIPPING)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss

    def gaussian_likelihood(self, actions, pred): # for keras custom loss
        log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        pre_sum = -0.5 * (((actions-pred)/(K.exp(log_std)+1e-8))**2 + 2*log_std + K.log(2*np.pi))
        return K.sum(pre_sum, axis=1)
    
    def gaussian_likelihood_numpy(self, action, pred, log_std):
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
        pre_sum = -0.5 * (((action-pred)/(np.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi)) 
        return np.sum(pre_sum, axis=1)

    def predict(self, state):
        return self.Actor.predict(state)
    
    def act(self, state):
        state = np.expand_dims(state, axis=0)
        pred = self.Actor.predict(state)
        low, high = -1.0, 1.0 # -1 and 1 are boundaries of tanh
        action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
        action = np.clip(action, low, high)
        
        logp_t = self.gaussian_likelihood_numpy(action, pred, self.log_std)

        return action, logp_t
    
    def mutate(self):
        for layer in self.Actor.layers:
            if len(layer.get_weights()) > 0:
                new_weights = [
                    layer.get_weights()[0] + np.random.normal(0, 0.15, layer.get_weights()[0].shape),
                    layer.get_weights()[1] + np.random.normal(0, 0.15, layer.get_weights()[1].shape)
                ]
                layer.set_weights(new_weights)

class DiscreteActorModel:
    def __init__(self, input_shape, action_space, lr, optimizer):

        X_input = Input(input_shape)
        self.action_space = action_space

        X = Conv2D(64,(1,1), activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Conv2D(32,1, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)

        #X = Conv2D(128,1, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        #X = Conv2D(64,1, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Flatten()(X)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Flatten()(X)
        output = Dense(self.action_space, activation="softmax")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))
        
        self.log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        self.std = np.exp(self.log_std)

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1+self.action_space:], y_true[:, 1:1+self.action_space]
        LOSS_CLIPPING = 0.3
        ENTROPY_LOSS = 0.0
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy
        return total_loss

    def predict(self, state):
        return self.Actor.predict(state)
    
    def act(self, state):
        # Use the network to predict the next action to take, using the model

        state = np.expand_dims(state, axis=0)
        prediction = self.Actor.predict(state)[-1]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

    def mutate(self):
        for layer in self.Actor.layers:
            if len(layer.get_weights()) > 0:
                new_weights = [
                    layer.get_weights()[0] + np.random.normal(0, 0.03, layer.get_weights()[0].shape),
                    layer.get_weights()[1] + np.random.normal(0, 0.03, layer.get_weights()[1].shape)
                ]
                layer.set_weights(new_weights)


class CriticModel:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        old_values = Input(shape=(1,))

        V = Conv2D(64,(1,1), activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        V = Conv2D(32,1, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)

        #V = Conv2D(128,1, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        #X = Conv2D(64,1, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        V = Flatten()(V)
        V= Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        V = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        V= Flatten()(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=[X_input, old_values], outputs = value)
        self.Critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
            return value_loss
        return loss

    def predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])



def row_col(position, columns):
    return position // columns, position 

def adjacent_positions(position: int, columns: int, rows: int):
    return [
        translate(position, action, columns, rows)
        for action in Action
    ]


def min_distance(position, food, columns):
    row, column = row_col(position, columns)
    return min(
        abs(row - food_row) + abs(column - food_column)
        for food_position in food
        for food_row, food_column in [row_col(food_position, columns)]
    )

def translate(position, direction, columns, rows):
    row, column = row_col(position, columns)
    row_offset, column_offset = direction.to_row_col()
    row = (row + row_offset) % rows
    column = (column + column_offset) % columns
    return row * columns + column


class GreedyAgentCustom:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.last_action = None

    def call(self, state, index):
        
        observation = state[0].observation
        rows, columns = self.configuration.rows, self.configuration.columns

        food = observation.food
        geese = observation.geese
        geese_idx = index
        opponents = [
            goose
            for index, goose in enumerate(geese)
            if index != geese_idx and len(goose) > 0
        ]

        # Don't move adjacent to any heads
        head_adjacent_positions = {
            opponent_head_adjacent
            for opponent in opponents
            for opponent_head in [opponent[0]]
            for opponent_head_adjacent in adjacent_positions(opponent_head, columns, rows)
        }
        # Don't move into any bodies
        bodies = {position for goose in geese for position in goose}

        # Move to the closest food
        position = geese[geese_idx][0]
        actions = {
            action: min_distance(new_position, food, columns)
            for action in Action
            for new_position in [translate(position, action, columns, rows)]
            if (
                new_position not in head_adjacent_positions and
                new_position not in bodies and
                (self.last_action is None or action != self.last_action.opposite())
            )
        }

        action = min(actions, key=actions.get) if any(actions) else choice([action for action in Action])
        self.last_action = action
        
        return GLOBAL_ACTIONS.index(action.name)