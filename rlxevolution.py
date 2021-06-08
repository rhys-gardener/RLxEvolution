import os
from re import I
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # -1:cpu, 0:first gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import random
import gym
import glob
import pylab
import numpy as np
import pandas as pd
import tensorflow as tf
#tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
tf.compat.v1.disable_eager_execution() # usually using this for fastest performance
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras import backend as K
import copy

import imageio

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from agents import ContinuousActorModel, DiscreteActorModel, CriticModel, GreedyAgentCustom

from EnvWrapper import EnvWrap

import multiprocessing as mp
import time

import math

tf.config.threading.set_intra_op_parallelism_threads(1)
mp.set_start_method('spawn', force=True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass


class PlaybackMemory:
    # Create an object to handle playback memory, allowing genetic memory
    def __init__(self):
        self.states_mem = []
        self.next_states_mem = []
        self.actions_mem = []
        self.rewards_mem = []
        self.dones_mem = []
        self.preds_mem = []

    def reset(self):
        self.states_mem = []
        self.next_states_mem = []
        self.actions_mem = []
        self.rewards_mem = []
        self.dones_mem = []
        self.preds_mem = []


    

class RLxEvolution:
    #An algorithm combining evolutionary methods with RL
    def __init__(self, env, population, model_name=""):
        
        # Initialization
        # Environment and PPO parameters
        self.env_name = env.env_name       
        self.Env = env
        # Parameters for evolution
        self.generation = 0
        self.GENERATION = 9999 #max generations
        self.population = population
        self.holdout = max(1, int(0.1 * self.population))

        self.discrete = self.Env.discrete
        self.action_size = self.Env.action_size


        self.state_size = self.Env.observation_space

        
        self.EPISODES = 200 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.0003 # Recommended set at max 0.001 
        self.epochs = 64 # training epochs
        self.shuffle = True
        self.TRAINING_BATCH = 10
        #self.optimizer = RMSprop
        self.optimizer = Adam

        self.replay_count = 0

        self.TRAINING_GAMES = 20000
        self.training_games = 0

        self.EVALUATION_GAMES = 10
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots
        # Create initial population 
        
      #  self.actors = self.create_initial_population(self.population)
        
        # Just the one critic for now
        print('state_size: ', self.state_size)
      #  self.Critic = CriticModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        
       # self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
      #  self.Critic_name = f"{self.env_name}_PPO_Critic.h5"
        #self.load() # uncomment to continue training from old weights

        # do not change bellow
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)
        self.num_actors = 4
        self.score_mem =         [0 for _ in range(self.num_actors)]
        self.states_mem =        []
        self.next_states_mem =   []
        self.actions_mem =       []
        self.rewards_mem =       []
        self.dones_mem =         []
     #   self.preds_mem =       [[] for _ in range(num_actors)]
        self.preds_mem =       []
        

    def create_initial_population(self, population=10):
        actors = []
        for i in range(0, population):
            if self.discrete:
                Actor = DiscreteActorModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
            else:
                Actor = ContinuousActorModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
            actors.append(Actor)
        return actors


    def gaussian_likelihood(self, action, pred, log_std):
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
        pre_sum = -0.5 * (((action-pred)/(np.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi)) 
        return np.sum(pre_sum, axis=1)

    def discount_rewards(self, reward):#gaes is better
        # Compute the gamma-discounted rewards over an episode
        # We apply the discount and normalize it to avoid big variability of rewards
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8) # divide by standard deviation
        return discounted_r

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.97, normalize=True):

        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actor, Critic, actions, rewards, dones, next_states, logp_ts):
        # reshape memory to appropriate shape for training
      #  print(actor)
     #   for idx, state in enumerate(states):
      #      print(f'state {idx}, shape{state.shape}')
        states = np.stack(states)
        next_states = np.stack(next_states)
        actions = np.stack(actions)
        logp_ts = np.stack(logp_ts)
        # Get Critic network predictions 
        values = Critic.predict(states)
        next_values = Critic.predict(next_states)

        # Compute discounted rewards and advantages
        #discounted_r = self.discount_rewards(rewards)
        #advantages = np.vstack(discounted_r - values)
      #  print(f'rewards shape: {len(rewards)}')
      #  print(f'dones shape: {len(dones)}')
      #  print(f'values: {len(values)}')
      #  print(f'next_values: {len(next_values)}')

        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom loss function we unpack it
        advantage_list = [advantage[0] for advantage in advantages]
        if len(actions.shape) > 2:
            actions_list = [tuple(map(tuple, action))[0] for action in actions]
        else:
            actions_list = [action for action in actions]


        y_true = np.hstack([advantages, actions, logp_ts])

        y_pred_test = actions_list

        # training Actor and Critic networks
        a_loss = actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0,batch_size=512, shuffle=self.shuffle)
        c_loss = Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0,batch_size=512, shuffle=self.shuffle)

        # calculate loss parameters (should be done in loss, but couldn't find working way how to do that with disabled eager execution)
        if not self.discrete:
            pred = actor.Actor.predict(states)
            log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
            logp = self.gaussian_likelihood(actions, pred, log_std)
            approx_kl = np.mean(logp_ts - logp)
            approx_ent = np.mean(-logp)

 #       self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
 #       self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
 #       if not self.discrete:
 #           self.writer.add_scalar('Data/approx_kl_per_replay', approx_kl, self.replay_count)
 #           self.writer.add_scalar('Data/approx_ent_per_replay', approx_ent, self.replay_count)
  #      self.replay_count += 1
    

    def create_new_weights(self, Model1, Model2):
        """ if self.discrete:
            child = DiscreteActorModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        else:
            print('continuous')
            child = ContinuousActorModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)"""
        new_weights = []
        for i in range(len(Model1.Actor.layers)):
            if len(Model1.Actor.layers[i].get_weights()) > 0:

                weights1 = Model1.Actor.layers[i].get_weights()[0]
                weights2 = Model2.Actor.layers[i].get_weights()[0]
                bias1 = Model1.Actor.layers[i].get_weights()[1]
                bias2 = Model2.Actor.layers[i].get_weights()[1]                
                #weights_pass = np.random.rand(1, weights1.shape[1]) < 0.5 
                weights_pass = np.random.randint(2, size=weights1.shape)
                bias_pass = np.random.randint(2, size=bias1.shape)
                #new_weights.append(weights_pass * weights1 + ~weights_pass * weights2)
                new_weights.append(weights1)
                new_weights.append(bias1)

               # new_bias = bias_pass * bias1  + ~bias_pass * bias2
               # child.Actor.layers[i].set_weights([new_weights, bias1]) 
        #child.mutate()
        return new_weights
    

    def generate_child(self, new_weights, mutate):
        if self.discrete:
            child = DiscreteActorModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        else:
            print('continuous')
            child = ContinuousActorModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        child.Actor.set_weights(new_weights)
        if mutate > 1:
            child.mutate()
        return child
 
    def load(self):
        self.Actor.Actor.load_weights(self.Actor_name)
        self.Critic.Critic.load_weights(self.Critic_name)

    def save(self):
        self.Actor.Actor.save_weights(self.Actor_name)
        self.Critic.Critic.save_weights(self.Critic_name)
    
    def save_defined_actor(self, actor, actor_name, critic, critic_name):
        actor.Actor.save_weights(actor_name)
        critic.Critic.save_weights(critic_name)

    pylab.figure(figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)

    
    def PlotModel(self, score, episode, save=True):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig(self.env_name+".png")
            except OSError:
                pass
        # saving best models
        if self.average_[-1] >= self.max_average and save:
            self.max_average = self.average_[-1]
            self.save()
            SAVING = "SAVING"
            # decreaate learning rate every saved model
            #self.lr *= 0.99
            #K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            #K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
        else:
            SAVING = ""

        return self.average_[-1], SAVING
    

    def run_batch(self, actor):
        self.episode = 0
        self.Env.env.reset()
        state = self.Env.get_state()
        #state = np.reshape(state, [1, self.state_size[0]])
        done, score, SAVING = False, 0, ''
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            for t in range(self.TRAINING_BATCH):
                #self.env.render()
                # Actor picks an action
                if self.discrete:
                    action, action_onehot, prediction = actor.act(state)
                    next_state, reward, done, _ = self.Env.step(action)
                else:
                    action, prediction = actor.act(state)
                    next_state, reward, done, _ = self.Env.step(action[0])

                # Memorize (state, action, reward) for training
                states.append(state)
                #next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                next_states.append(next_state)

                if self.discrete: 
                    actions.append(action_onehot)
                else:
                    actions.append(action)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                # Update current state
                #state = np.reshape(next_state, [1, self.state_size[0]])
                state = next_state
                score += reward
                print('Done', done)
                if done:
                    self.episode += 1
                    #average, SAVING = self.PlotModel(score, self.episode)
                    if self.episode%10==0:
                        #print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                        print(f"episode: {self.episode}/{self.EPISODES}, score={score}")
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)
                    self.Env.env.reset()
                    state, done, score, SAVING = self.Env.get_state(), False, 0, ''
                    #state = np.reshape(state, [1, self.state_size[0]])

            self.replay(states, actor, actions, rewards, dones, next_states, predictions)
            if self.episode >= self.EPISODES:
                #self.replay(states, actor, actions, rewards, dones, next_states, predictions)
                break
                
        #self.env.close()
        return score
    
    def run_competition_training(self, actors, critic, evaluate): 

        """Train multiple models against eachother in a game, e.g. for kaggle"""
        episode = 0
        Env = EnvWrap('kaggle','hungry_geese')
        Env.reset()
        state = Env.get_state()
        #state = np.reshape(state, [1, self.state_size[0]])
        training_games=0
        if evaluate==True:
            TRAINING_GAMES = 1
        else:
            TRAINING_GAMES = self.TRAINING_GAMES
        
        num_actors = len(actors)
        score_mem = [0 for _ in range(self.num_actors)]
        image_path = f'images/generation{self.generation}/'

        agent_mems = [PlaybackMemory() for _ in range (num_actors)]


        print(f'initial score mem: {score_mem}')
        for b in range(self.TRAINING_BATCH):
            for t in range(TRAINING_GAMES):
                done, score, SAVING = False, 0, ''
                greedy_agents = [
                    GreedyAgentCustom(Env.env.configuration),
                    GreedyAgentCustom(Env.env.configuration),
                    GreedyAgentCustom(Env.env.configuration),
                    GreedyAgentCustom(Env.env.configuration)
                ]

                prev_reward = [0,0,0,0]
                current_game_rewards = [[] for _ in range(num_actors)]
                step = 0
                images = []
                temp_states = []
                while done == False:

                    current_actions = []
                    current_actions_onehot = []
                    temp_predictions = []
                    for actor_idx, actor in enumerate(actors):
                        state = Env.get_geese_observation(actor_idx, Env.env.state)
                        temp_states.append(state)
                        # agent_mems[actor_idx].states_mem.append(state)
                        if len(Env.env.state[0].observation.geese[actor_idx]) > 0:
                            


                            probability_actor = math.exp(-(self.generation+1)/30)

                            proba = [1 - probability_actor, probability_actor]
                            actor_flag = random.choices(population=[True,False], weights=proba ,k=1)[0]
                            #actor_flag = True
                            if (actor_flag == True):
                                #use trained actor
                                action, prediction = actor.act(state)
                            elif (len(Env.env.state[0].observation.geese[actor_idx]) > 0) and (evaluate==False):
                                #use greedy agent to choose action
                                action = greedy_agents[actor_idx].call(Env.env.state, actor_idx)
                                prediction = np.empty([self.action_size])
                                prediction.fill(0.0025)

                                prediction[action] = 0.99
                            else:
                                action, prediction = actor.act(state)

                            #also see what action the greedy agent comes up with
        
                            action_onehot = np.zeros([self.action_size])
                            action_onehot[action] = 1
                            current_actions.append(action)
                            current_actions_onehot.append(action_onehot)
                            temp_predictions.append(prediction)


                        else:
                            current_actions.append(4)
                            current_actions_onehot.append([0,0,0,0])
                            temp_predictions.append(0)

                    # print(f'current_actions: {current_actions}')
                    next_states, rewards, dones, done = Env.multistep(current_actions)
                    # print(f'len next states: {len(next_states)}')
                    for i in range(0, len(next_states)):
                        if current_actions[i] != 4:
                            agent_mems[i].states_mem.append(temp_states[i])

                            agent_mems[i].next_states_mem.append(next_states[i])
                            agent_mems[i].rewards_mem.append(rewards[i])
                            agent_mems[i].dones_mem.append(dones[i])
                            current_game_rewards[i].append(rewards[i])
                            agent_mems[i].actions_mem.append(current_actions_onehot[i])
                            agent_mems[i].preds_mem.append(temp_predictions[i])

                    if done:

                        scores = [sum(reward) for reward in current_game_rewards]
                        training_games += 1
                        print(f"Training game: {training_games}/{self.TRAINING_GAMES}, score={scores}")
                        Env.reset()           
                        for ix, score in enumerate(scores):
                            score_mem[ix] = score_mem[ix] + score

            for idx, actor in enumerate(actors):
                if len(agent_mems[idx].states_mem) > 0:
                    self.replay(agent_mems[idx].states_mem, actor,critic, agent_mems[idx].actions_mem, agent_mems[idx].rewards_mem, agent_mems[idx].dones_mem, agent_mems[idx].next_states_mem, agent_mems[idx].preds_mem)
                else:
                    print(idx)
                    print('no data')
            for mem in agent_mems:
                mem.reset()
                
            Env.reset()

        # cleanup
        del agent_mems                      
        return True

    
    def run_competition_evaluation(self, actors, vis):
        episode = 0
        Env = EnvWrap('kaggle','hungry_geese')
        Env.reset()
        state = Env.get_state()
        #state = np.reshape(state, [1, self.state_size[0]])
       
        num_actors = len(actors)
        score_mem = [[] for _ in range(self.num_actors)]
        image_path = f'images/generation{self.generation}/'
        game_image_path = image_path

        for t in range(self.EVALUATION_GAMES):
            done, score, SAVING = False, 0, ''
            prev_reward = [0,0,0,0]
            current_game_rewards = [[] for _ in range(num_actors)]
            step = 0
            images = []
            while done == False:
                if vis == True:
                    render = Env.env.render(mode="ansi")
                    game_image_path = image_path + f'game{t}/'
                    # os.makedirs(game_image_path, exist_ok=True)
                    Path(game_image_path).mkdir(parents=True, exist_ok=True)
                    filename = f'{game_image_path}/image{step:03d}.png'
                    image = Image.new(mode = "RGB", size = (300,300), color = "white")
                    draw = ImageDraw.Draw(image)
                    draw.text((10,10), render, fill=(0,0,0))
                    image.save(filename)
                    step += 1

                current_actions = []
                for actor_idx, actor in enumerate(actors):
                    state = Env.get_geese_observation(actor_idx, Env.env.state)
                    if len(Env.env.state[0].observation.geese[actor_idx]) > 0:
                        action, prediction = actor.act(state)
                        action_onehot = np.zeros([self.action_size])
                        action_onehot[action] = 1
                        current_actions.append(action)
                    else:
                        current_actions.append(4)
                
                next_states, rewards, dones, done = Env.multistep(current_actions)

                for i in range(0, len(next_states)):
                    current_game_rewards[i].append(rewards[i])

                if done:
                    scores = [sum(reward) for reward in current_game_rewards]
                    print(f"Evaluation game: {t}/{self.EVALUATION_GAMES}, score={scores}")
                    Env.reset()           
                    for ix, score in enumerate(scores):
                        score_mem[ix].append(score)

                
                            #create gif
                if vis == True:
                    png_dir = game_image_path
                    images = []
                    for file_name in sorted(os.listdir(png_dir)):
                        if file_name.endswith('.png'):
                            file_path = os.path.join(png_dir, file_name)
                            images.append(imageio.imread(file_path))
                    kargs = { 'duration': 0.5 }
                    imageio.mimsave(f'{game_image_path}/movie.gif', images, **kargs)

        return score_mem
                    
    def chunks(self, l, n):
        n = max(1, n)
        return (l[i:i+n] for i in range(0, len(l), n))


    def create_new_population(self, model1, model2):
        new_weights_list = []
        new_weights_list.append(model1.Actor.get_weights())
        new_population = []
        for i in range(self.population-1):
            new_weights = self.create_new_weights(model1, model2)
            new_weights_list.append(new_weights)
            #new_population.append(offspring)

        # Create new population and set weights
        for i in range(0, len(new_weights_list)):
            child = self.generate_child(new_weights_list[i], i)
            new_population.append(child)
        return new_population


    def run_parallel_competitions(self, actor_paths, critic_path):
      #  actor_paths = pairing[0]
      #  critic_path = pairing[1]
        actors = []
        for actor_path in actor_paths:
            actor = DiscreteActorModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)

            actor.Actor.load_weights(actor_path)
            actors.append(actor)
        critic = CriticModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        critic.Critic.load_weights(critic_path)
        self.run_competition_training(actors,critic, evaluate=False)
        training_scores = self.run_competition_evaluation(actors, vis=False)
        scores = []
        for score_list in training_scores:
            scores.append(np.median(score_list))



        actors = [actors[x] for x in np.argsort(scores)[::-1]]
        best_actor_idx = np.argsort(scores)[::-1][0]
        best_actor = actors[best_actor_idx]
        best_score = max(scores)
        #save actor
        id = random.randint(0, 100000)
        best_actor_path = f'winning_actors/actor{id}.h5'
        best_actor.Actor.save_weights(best_actor_path)
        #save associated critic
        critic_path_end = f'critics/critic{id}.h5'
        critic.Critic.save_weights(critic_path_end)
        print('end: ', (best_actor_path, best_score))
        return (best_actor_path, best_score, id)
    
    def parallel_test(self, score):
        return score

    def run_parallel_test(self):
        pool = mp.Pool(4)
        results = pool.map_async(self.parallel_test, [1]).get()
        print(results)

    def run_competitive_evolution(self, entrants=4, load_champion=False, champion_name = None, critic_name = None):
        """
        There may be the opportunity to train models against itself. 
        Particularly useful with kaggle environments
        """
        loop = False
        scores_df = pd.DataFrame(columns=['generation','best_score', 'best_actor'])
        actors = []
        if load_champion == True:
            if self.discrete:
                champion = DiscreteActorModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
            else:
                champion = ContinuousActorModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)

            champion.Actor.load_weights(champion_name)

            critic = CriticModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
            critic.Critic.load_weights(critic_name)

            actors = self.create_new_population(champion, champion)

        else:
            actors = self.create_initial_population(self.population)
            critic = CriticModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        
        # Just the one critic for now
        print('state_size: ', self.state_size)
        

        while self.generation < self.GENERATION:
            print('GENERATION:', self.generation)

            # Save actors - needed for parallel
            actor_paths = []
            
            for i, actor in enumerate(actors):
                modelpath = f'working_models/actor{i}.h5'
                actor.Actor.save_weights(modelpath)
                actor_paths.append(modelpath)

            critic_path = f'working_models/critic.h5'
            critic.Critic.save_weights(critic_path)

            K.clear_session()

            best = []
            best_paths = []
            best_scores = []
            ids = []
            random.shuffle(actors)
            actor_pairings = [(actor_paths[x:x+4], critic_path) for x in range(0, len(actor_paths), 4)]

            pool = mp.Pool(8)
            results = pool.starmap_async(self.run_parallel_competitions, actor_pairings).get()
            pool.close()

            for result in results:
                best_paths.append(result[0])
                best_scores.append(result[1])
                ids.append(result[2])
            # select best 4 from list of best
            print(f'best scores: {best_scores}')
            best_paths = [best_paths[x] for x in np.argsort(best_scores)[::-1]][:4]
            while len(best_paths) < 4:
                best_paths.append(best_paths[0])

            for actor_path in best_paths:
                actor = DiscreteActorModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)

                actor.Actor.load_weights(actor_path)
                best.append(actor)
            

            final_scores_eval = self.run_competition_evaluation(best,vis=True)
            final_scores = []
            for score_list in final_scores_eval:
                final_scores.append(np.median(score_list))
            print('Final scores of generation: ', final_scores)
            champion_actor_idx = np.argsort(final_scores)[::-1][0]



            champion_actor = [best[x] for x in np.argsort(final_scores)[::-1]][0]
            runner_up = [best[x] for x in np.argsort(final_scores)[::-1]][1]

            scores_df.loc[self.generation] = [self.generation, max(final_scores), np.argsort(final_scores)[::-1][0]] 
            scores_df.to_csv('results.csv')
            #save model
            actor_name = f'models/champion_actor_{self.generation}.h5'
            critic_name = f'models/champion_critic_{self.generation}.h5'
            critic = CriticModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
            critic.Critic.load_weights(f'critics/critic{ids[champion_actor_idx]}.h5')
            self.save_defined_actor(champion_actor, actor_name,critic, critic_name)

            # Create new population
            new_population = []

            new_weights_list = []
            new_weights_list.append(champion_actor.Actor.get_weights())
            for i in range(self.population-1):
                new_weights = self.create_new_weights(champion_actor, runner_up)
                new_weights_list.append(new_weights)
                #new_population.append(offspring)
            
            #Get critic weights, as we need to clear the tf backend (otherwise memory issues on large population size?)
            critic_weights = critic.Critic.get_weights()
            del critic
            tf.keras.backend.clear_session()
            
            critic = CriticModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
            critic.Critic.load_weights(f'critics/critic{ids[champion_actor_idx]}.h5')

            # Create new population and set weights
            for i in range(0, len(new_weights_list)):
                child = self.generate_child(new_weights_list[i], i)
                new_population.append(child)
            self.generation +=1
            actors = new_population
        
            # Clear out the working directories for critics and winning_actors
            critic_files = glob.glob('critics/*.h5')
            actor_files = glob.glob('winning_actors/*.h5')
            for f in critic_files:
                try:
                    os.remove(f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))
            for f in actor_files:
                try:
                    os.remove(f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))
            



                

    def run_evolution(self):
        print(len(self.actors))
        scores_df = pd.DataFrame(columns=['generation','best_score', 'best_actor'])
        while self.generation < self.GENERATION:
            scores = []
            print(f"generation: {self.generation}")
            for actor in self.actors:           
                self.run_batch(actor)
                score = self.evaluate(actor, test_episodes=3)
                scores.append(score)
            print('testing')
            print(scores)
            print(np.argsort(scores)[::-1])
            print([x for x in np.argsort(scores)[::-1]])
            self.actors = [self.actors[x] for x in np.argsort(scores)[::-1]]
            
            # best_organism = self.actors[0]
            best_score = max(scores)
            best_actor_idx = np.argsort(scores)[::-1][0]
            scores_df.loc[self.generation] = [self.generation, best_score, best_actor_idx] 
            scores_df.to_csv('results.csv')
            print('best score: ', best_score)
            new_population = []

            #new_population.append(self.actors[0])
            #tf.keras.backend.clear_session()
           # new_population.append(self.actors[1])
            new_weights_list = []
            new_weights_list.append(self.actors[0].Actor.get_weights())
            for i in range(self.population-1):
                parent_1_idx = 0
                #parent_2_idx = min(self.population - 1, int(np.random.exponential(self.holdout)))
                parent_2_idx = 1
                print(f'parents: {parent_1_idx}, {parent_2_idx}')
                new_weights = self.create_new_weights(self.actors[parent_1_idx], self.actors[parent_2_idx])
                new_weights_list.append(new_weights)
                #new_population.append(offspring)
            
            critic_weights = self.Critic.Critic.get_weights()

            tf.keras.backend.clear_session()
            del self.Critic
            self.Critic = CriticModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
            self.Critic.Critic.set_weights(critic_weights)

            # Create new population and set weights
            for i in range(0, len(new_weights_list)):
                print('child: ', i)
                child = self.generate_child(new_weights_list[i], i)
                new_population.append(child)
            self.generation +=1
            self.actors = new_population
            


    def evaluate(self, actor, test_episodes = 100):#evaluate
        score = 0
        for e in range(test_episodes):
            self.Env.reset()
            state = self.Env.get_state()
            #state = np.reshape(state, [1, self.state_size[0]])
            done = False

            while not done:
                out = self.Env.env.render(mode="ansi")
                print(out)
                 # Actor picks an action
               # action = actor.act(state)[0]
                                # Actor picks an action
                if self.discrete:
                    action = actor.act(state)[0]
                    state, reward, done, _ = self.Env.step(action)
                else:
                    action = actor.act(state)[0]
                    state, reward, done, _ = self.Env.step(action[0])

                #state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    #average, SAVING = self.PlotModel(score, e, save=False)
                    print("evaluation episode: {}/{}, score: {}".format(e, test_episodes, score))
                    break
        #self.env.close()
        print('total score: ', score)
        return score

if __name__ == "__main__":
    # newest gym fixed bugs in 'BipedalWalker-v2' and now it's called 'BipedalWalker-v3'
    print('yay')
    #env_name = 'BipedalWalker-v3'
    #env_name = "CartPole-v1"
    #env_name = 'LunarLanderContinuous-v2'
    env_name = 'AirRaid-v0'
    #env = EnvWrap('gym',env_name)
    env=EnvWrap('kaggle','hungry_geese')



    agent = RLxEvolution(env, 4)
    #agent.run_evolution() # train as PPO
    champion = 'models/champion_actor_24.h5'
    coach = 'models/champion_critic_24.h5'
    agent.run_competitive_evolution()
    #agent.run_competitive_evolution(load_champion=True, champion_name = champion, critic_name = coach)
    #agent.run_multiprocesses(num_worker = 16)  # train PPO multiprocessed (fastest)
    #agent.evaluate(agent.actors[0], 10)