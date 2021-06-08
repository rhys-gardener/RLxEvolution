# Kaggle training
# This is a slightly different version of the wrapper to use with the RAY RLlib training
from collections import Counter
from math import sqrt
import gym
import kaggle_environments as kaggle
import numpy as np

import logging



from gym.spaces import Discrete, Box

from ray.rllib.env.multi_agent_env import MultiAgentEnv

class HungryGeeseKaggle(MultiAgentEnv):
    """A class to contain environments, so that a learning algorithm can just access the observation space and actions pace
    I wouldn't have bothered but kaggle environments need some additional processing around the observation space
    And I might as well create this in case I decide to make some environments in future..."""
    def __init__(self, config):
        
        self.env = kaggle.make("hungry_geese")
        self.num_agents = 4
        self.env.reset(self.num_agents)
        self.rows = self.env.configuration.rows
        self.columns = self.env.configuration.columns
        #self.observation_space = np.array((self.rows, self.columns, 3))
        self.observation_space = Box(low=-100, high=1000, shape=(self.rows, self.columns, 3), dtype=np.float32)
        # self.observation_space = self.rows * self.columns
        self.action_space = Discrete(4)
        self.discrete = True
        self.actions = ['NORTH','SOUTH','WEST','EAST', '']
        self.prev_head_locations = [0,0,0,0]
        self.food_history = [0,0,0,0]
        self.names = ['geese1','geese2','geese3','geese4']
        logging.basicConfig(filename='logging2.log', level=logging.DEBUG)

                
    def reset(self):
        self.env.reset(num_agents=self.num_agents)
        states = {}
        for i in range(0, self.num_agents):
            states[self.names[i]] = self.get_geese_observation(i, self.env.state)
        logging.debug(f'initial states: {states}')
        return states
        
    '''
    def step(self, action, agent=0):

        status = self.env.step([self.actions[action]])
        state = self.get_geese_observation(agent, self.env.state)
        reward = status[agent]['reward']
        if status[agent]['status']=='DONE':
            done = True
        else:
            done = False
        return state, reward, done, 1
    ''' 
    
    def step(self, action_dict):
        """Useful if the environment accepts multiple actions simultaneously"""

        action1 = action_dict.get('geese1', 0)
        action2 = action_dict.get('geese2', 0)
        action3 = action_dict.get('geese3', 0)
        action4 = action_dict.get('geese4', 0)

        actions = [action1, action2, action3, action4]

        
        done = False
        prev_status = self.env.state
        for i in range(0, self.num_agents):
            old_board = self.get_geese_observation(i, prev_status)
            old_geese_loc = self.get_geese_coord(old_board)

            if len(old_geese_loc) > 0:
                self.prev_head_locations[i] = old_geese_loc[0]


        status = self.env.step([self.actions[action] for action in actions])
        
        running = False
        next_states = {}
        rewards = {}
        dones = {}
        for i in range(0, self.num_agents):

            next_states[self.names[i]] = self.get_geese_observation(i, self.env.state)
            reward = self.reward_geese(prev_status, status, i)
            #rewards.append(status[i]['reward'])
            rewards[self.names[i]] = reward
            if status[i]['status']=='DONE':
                dones[self.names[i]] = True
            else:
                dones[self.names[i]] = False
            if status[i]['status']=='ACTIVE':
                running = True

        '''
        if False not in dones:
            done = True
        else:
            done = False
        '''
        if running == False:
            dones['__all__'] = True
        else:
            dones['__all__'] = False

        logging.debug(f'Rewards: {rewards}')
        return next_states, rewards, dones, {}
    
    def reward_geese(self, prev_status, status, geese):

        step = status[0].observation.step
        reward = status[geese]['reward']
        step_reward = 0
        old_length = len(prev_status[0].observation.geese[geese])
        new_length = len(status[0].observation.geese[geese])

        old_board = self.get_geese_observation(geese, prev_status)
        board = self.get_geese_observation(geese, self.env.state)


        old_geese_loc = self.get_geese_coord(old_board)
        geese_loc = self.get_geese_coord(board)

        old_food_loc = self.get_food_coord(old_board)
        food_loc = self.get_food_coord(board)

        enemy_geese_loc = self.get_enemy_geese_head_coord(board)
    #    print('testing')
    #    print(f'old food: {old_food_loc}, new_food: {food_loc}, old geese: {old_geese_loc}, new geese: {geese_loc}')
    #    print(f'enemy geese: {enemy_geese_loc}')


        old_distances = []
        new_distances = []
        move_reward = 0
        # Measure the distance to old food only - as new food pops up when eaten
        if (len(geese_loc) > 0) & (len(old_food_loc) > 0):
            old_distances = [self.get_distance_toroidal(old_geese_loc[0], food) for food in old_food_loc]
            new_distances = [self.get_distance_toroidal(geese_loc[0], food) for food in food_loc]
            #print(f'testing: old_distances: {old_distances}, new_distances: {new_distances}')

            old_min_distance = min(old_distances)
            new_min_distance = min(new_distances)
            if old_min_distance > new_min_distance:
                # Moved closer to a food
                move_reward = 100 / (new_min_distance + 1)
                #print('rewarded')
            else:
                #moved away
                move_reward = -20
                #print('punished')

        length_reward = 0
        food_reward = 0
        punish = 0

        # If the move kills the geese, then punish accordingly
        #if new_length == 0:
        #    punish = -20

        # Food reward is based on how quickly food was obtained
        if new_length > old_length:
            food_reward = 200 - 2*self.food_history[geese]
            self.food_history[geese] = 0
        else:
            self.food_history[geese] += 1
        
        #print(self.food_history)
        # Check whether the geese was adjacent to food and missed it

        #print(f'reward calc: reward: {reward}, step_reward {step_reward}, length {length_reward}')
        return step_reward + length_reward + food_reward + punish + move_reward
    

    def get_geese_coord(self, board):
        return self.get_coord_from_np_grid(board, 101)
    
    def get_food_coord(self, board):
        return self.get_coord_from_np_grid(board, 1000)
    
    def get_enemy_geese_head_coord(self, board):
        return self.get_coord_from_np_grid(board, -99)
   
    
    def get_coord_from_np_grid(self, grid, value):
        coords = []
        for i in range(0, len(np.where(grid==value)[0])):
            coords.append((np.where(grid==value)[0][i], np.where(grid==value)[1][i]))
        return coords
    

    def get_distance_toroidal(self, coord1, coord2):
        x1, y1 = coord1[0], coord1[1]
        x2, y2 = coord2[0], coord2[1]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx > 0.5*self.rows:
            dx = self.rows - dx
        
        if dy > 0.5*self.columns:
            dy = self.columns - dy

        return sqrt(dx*dx + dy*dy)

    
    def coordinates_adjacent_check(self, coord1, coord2):
        x1, y1 = coord1
        x2, y2 = coord2
        if x1==x2:
            if abs(y1 - y2) == 1:
                return True
            else:
                return False
        elif y2 == y1:
            if abs(x1 - x2) == 1:
                return True
            else:
                return False
        else:
            return False

    def get_coordinate(self, item, columns):
        (x, y) = divmod(item+1, columns)
        x = x
        y = y - 1
        return (x, y)
        
    
    def get_state(self, agent=0):
        if self.env_type=='gym':
            if self.env_name=='AirRaid-v0':
                return self.env.ale.getScreenRGB2()

            else:
                return self.env.state
        elif self.env_type=='kaggle':
            return self.get_geese_observation(agent, self.env.state)   
    
    def get_geese_observation(self, agent, state):
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

        game_board_self = np.zeros(self.rows*self.columns, None)
        game_board_enemy = np.zeros(self.rows*self.columns, None)
        game_board_food = np.zeros(self.rows*self.columns, None)


        for i, geese in enumerate(state[0].observation.geese):
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
                
        for food in state[0].observation.food:
            game_board_food[food] = 1000
        game_board_self = game_board_self.reshape([self.rows, self.columns])
        game_board_enemy = game_board_enemy.reshape([self.rows, self.columns])
        game_board_food = game_board_food.reshape([self.rows, self.columns])

        head = self.get_geese_coord(game_board_self)

        if len(head)==0:
            head = self.prev_head_locations[agent]
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
        




    