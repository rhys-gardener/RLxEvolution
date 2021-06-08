import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1:cpu, 0:first gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import random
import gym
import pylab
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboardX import SummaryWriter
#tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
tf.compat.v1.disable_eager_execution() # usually using this for fastest performance
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras import backend as K
import copy

from agents import ContinuousActorModel, DiscreteActorModel, CriticModel

from threading import Thread, Lock
from multiprocessing import Process, Pipe
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass


class RLxEvolution:
    #An algorithm combining evolutionary methods with RL
    def __init__(self, env_name, population, model_name=""):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = gym.make(env_name)
        # Parameters for evolution
        self.generation = 0
        self.GENERATION = 200 #max generations
        self.population = population
        self.holdout = max(1, int(0.1 * self.population))

        print(hasattr(self.env.action_space, 'n'))
        if hasattr(self.env.action_space, 'n'):
            self.discrete = True
            self.action_size = self.env.action_space.n
        else:
            self.discrete = False
            self.action_size = self.env.action_space.shape[0]
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 100 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.0003
        self.epochs = 20 # training epochs
        self.shuffle = True
        self.Training_batch = 100
        #self.optimizer = RMSprop
        self.optimizer = Adam

        self.replay_count = 0
        self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots
        # Create initial population 
        self.actors = self.create_initial_population(self.population)
        # Just the one critic for now
        self.Critic = CriticModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        
        self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
        self.Critic_name = f"{self.env_name}_PPO_Critic.h5"
        #self.load() # uncomment to continue training from old weights

        # do not change bellow
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)
    

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

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.98, lamda = 0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actor, actions, rewards, dones, next_states, logp_ts):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        logp_ts = np.vstack(logp_ts)
       
        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute discounted rewards and advantages
        #discounted_r = self.discount_rewards(rewards)
        #advantages = np.vstack(discounted_r - values)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom loss function we unpack it
        y_true = np.hstack([advantages, actions, logp_ts])
        #print(y_true)
        # training Actor and Critic networks
        a_loss = actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        # calculate loss parameters (should be done in loss, but couldn't find working way how to do that with disabled eager execution)
        if not self.discrete:
            pred = actor.Actor.predict(states)
            log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
            logp = self.gaussian_likelihood(actions, pred, log_std)
            approx_kl = np.mean(logp_ts - logp)
            approx_ent = np.mean(-logp)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        if not self.discrete:
            self.writer.add_scalar('Data/approx_kl_per_replay', approx_kl, self.replay_count)
            self.writer.add_scalar('Data/approx_ent_per_replay', approx_ent, self.replay_count)
        self.replay_count += 1
    

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
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score, SAVING = False, 0, ''
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            for t in range(self.Training_batch):
                #self.env.render()
                # Actor picks an action
                if self.discrete:
                    action, action_onehot, prediction = actor.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                else:
                    action, prediction = actor.act(state)
                    next_state, reward, done, _ = self.env.step(action[0])

                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
         
                if self.discrete: 
                    actions.append(action_onehot)
                else:
                    actions.append(action)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                # Update current state
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                if done:
                    self.episode += 1
                    #average, SAVING = self.PlotModel(score, self.episode)
                    if self.episode%10==0:
                        #print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                        print(f"episode: {self.episode}/{self.EPISODES}, score={score}")
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)

                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state = np.reshape(state, [1, self.state_size[0]])
            self.replay(states, actor, actions, rewards, dones, next_states, predictions)
            if self.episode >= self.EPISODES:
                #self.replay(states, actor, actions, rewards, dones, next_states, predictions)
                break
                
        #self.env.close()
        return score

    

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
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            
            while not done:
                self.env.render()
                 # Actor picks an action
                action = actor.act(state)[0]
                                # Actor picks an action
                if self.discrete:
                    action = actor.act(state)[0]
                    state, reward, done, _ = self.env.step(action)
                else:
                    action = actor.act(state)[0]
                    state, reward, done, _ = self.env.step(action[0])

                state = np.reshape(state, [1, self.state_size[0]])
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
    env_name = 'BipedalWalker-v3'
    #env_name = "CartPole-v1"
    #env_name = 'LunarLanderContinuous-v2'
    agent = RLxEvolution(env_name, 10)
    agent.run_evolution() # train as PPO
    #agent.run_multiprocesses(num_worker = 16)  # train PPO multiprocessed (fastest)
    #agent.evaluate(agent.actors[0], 10)