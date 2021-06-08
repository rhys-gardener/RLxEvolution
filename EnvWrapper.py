# Kaggle training
from collections import Counter
from math import sqrt
import gym
import kaggle_environments as kaggle
import numpy as np

FOOD_VALUE = 100
SELF_HEAD_VALUE = 50
SELF_BODY_VALUE = 51
ENEMY_HEAD_VALUE = 10
ENEMY_BODY_VALUE = 11

class EnvWrap:
    """A class to contain environments, so that a learning algorithm can just access the observation space and actions pace
    I wouldn't have bothered but kaggle environments need some additional processing around the observation space
    And I might as well create this in case I decide to make some environments in future..."""
    def __init__(self, env_type, env_name, config=None):
        self.env_name = env_name
        self.env_type = env_type
        if self.env_type == 'gym':
            self.env = gym.make(env_name)
            self.observation_space = self.env.observation_space.shape
            if hasattr(self.env.action_space, 'n'):
                self.discrete = True
                self.action_size = self.env.action_space.n
            else:
                self.discrete = False
                self.action_size = self.env.action_space.shape[0]
            
        
        elif env_type=='kaggle':
            if env_name == 'hungry_geese':
                self.env = kaggle.make("hungry_geese")
                self.num_agents = 4
                self.env.reset(self.num_agents)
                self.rows = self.env.configuration.rows
                self.columns = self.env.configuration.columns
                self.observation_space = (self.rows, self.columns, 3)
                # self.observation_space = self.rows * self.columns
                self.action_size = 4
                self.discrete = True
                self.actions = ['NORTH','SOUTH','WEST','EAST', '']
                self.prev_head_locations = [0,0,0,0]
                self.food_history = [0,0,0,0]
                
        else:
            print("I don't know what to do :( ")
            return None
    

    def reset(self):
        if self.env_type=="gym":
            self.env.reset()
        elif self.env_type=='kaggle':
            self.env.reset(num_agents=self.num_agents)


    def step(self, action, agent=0):
        if self.env_type=='gym':
            return self.env.step(action)

        if self.env_type=='kaggle':

            status = self.env.step([self.actions[action]])
            state = self.get_geese_observation(agent, self.env.state)
            reward = status[agent]['reward']
            if status[agent]['status']=='DONE':
                done = True
            else:
                done = False
            return state, reward, done, 1
    
    def multistep(self, actions):
        """Useful if the environment accepts multiple actions simultaneously"""
        if self.env_type=='kaggle':
            done = False
            prev_status = self.env.state
            for i in range(0, self.num_agents):
                old_board = self.get_geese_observation(i, prev_status)
                old_geese_loc = self.get_geese_coord(old_board)

                if len(old_geese_loc) > 0:
                    self.prev_head_locations[i] = old_geese_loc[0]


            status = self.env.step([self.actions[action] for action in actions])
            next_states, rewards, dones = [], [], []

            running = False
            for i in range(0, self.num_agents):

                next_states.append(self.get_geese_observation(i, self.env.state))
                reward = self.reward_geese(prev_status, status, i)
                #rewards.append(status[i]['reward'])
                rewards.append(reward)
                if status[i]['status']=='DONE':
                    dones.append(True)
                else:
                    dones.append(False)
                if status[i]['status']=='ACTIVE':
                    running = True
            '''
            if False not in dones:
                done = True
            else:
                done = False
            '''
            if running == False:
                done = True
            else:
                done = False

            return next_states, rewards, dones, done
    
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
                move_reward = 10 / (new_min_distance + 1)
                #print('rewarded')
            else:
                #moved away
                move_reward = -2
                #print('punished')

        length_reward = 0
        food_reward = 0
        punish = 0

        # If the move kills the geese, then punish accordingly
        #if new_length == 0:
        #    punish = -20

        # Food reward is based on how quickly food was obtained
        if new_length > old_length:
            food_reward = 40 - self.food_history[geese]/2
            self.food_history[geese] = 0
        else:
            self.food_history[geese] += 1
        
        #print(self.food_history)
        # Check whether the geese was adjacent to food and missed it

        #print(f'reward calc: reward: {reward}, step_reward {step_reward}, length {length_reward}')
        return step_reward + length_reward + food_reward + punish + move_reward
    

    def get_geese_coord(self, board):
        return self.get_coord_from_np_grid(board, SELF_HEAD_VALUE)
    
    def get_food_coord(self, board):
        return self.get_coord_from_np_grid(board, FOOD_VALUE)
    
    def get_enemy_geese_head_coord(self, board):
        return self.get_coord_from_np_grid(board, ENEMY_HEAD_VALUE)
   
    
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
                for j, cell in enumerate(geese):
                    if j == 0:
                        game_board_self[cell] = SELF_HEAD_VALUE
                    else:
                        game_board_self[cell] = SELF_BODY_VALUE
            else:
                identify=-100
                for j, cell in enumerate(geese):
                    if j == 0:
                        game_board_enemy[cell] = ENEMY_BODY_VALUE
                    else:
                        game_board_enemy[cell] = ENEMY_HEAD_VALUE
                
        for food in state[0].observation.food:
            game_board_food[food] = FOOD_VALUE
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
        




    