import numpy as np
import matplotlib.pyplot as plt
import math
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import model_from_json
from PIL import ImageGrab
import cv2
import pyautogui
import sys
import time
from collections import namedtuple
from replaymemory import replaymemory
from game import game
from epsilongreedy import EpsilonGreedyStrategy

# hyper parameters
NUM_ACTIONS = 3
IMAGE_SIZE = (84,84,4)
STACK_IMAGE_SIZE = (1,84,84,4)
ACTION_SPACE_SIZE = 3
BATCH_SIZE = 16
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.001
EPS_DECAY = 0.001
TARGET_UPDATE = 2
MEMORY_SIZE = 10000
LEARNING_RATE = 1e-4

# one episode is one complete gameplay from start till end state(gameover state)
class dqn:
    def __init__(self):
        print('Initializing DQN Agent')
        self.current_step = 0
        self.reward = 0
        self.replaymemory = replaymemory(MEMORY_SIZE)
        self.strategy = EpsilonGreedyStrategy(EPS_START,EPS_END,EPS_DECAY)
        self.game = game()
        # model is for fitting. target_model, clone of main model for predictions
        self.model = self.createModel()
        self.target_model = self.createModel()
        self.target_model.set_weights(self.model.get_weights())

        '''self.model = self.loadModel('model')
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
        self.target_model = self.loadModel('target_model')'''

        self.experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

    def createModel(self):
        model = Sequential()

        model.add(Convolution2D(32, (8, 8),strides=(4, 4),activation='relu',input_shape=IMAGE_SIZE))
        model.add(Convolution2D(64, (4, 4),strides=(2, 2),activation='relu'))
        model.add(Convolution2D(64, (3, 3),strides=(1, 1),activation='relu'))

        model.add(Flatten())
        model.add(Dense(512))

        # ACTION_SPACE_SIZE = how many choices (3)
        model.add(Dense(ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
        return model

    def saveModel(self,model_name,model):
        model_json = model.to_json()
        modelpath = model_name+'/'+model_name+'.json'
        with open(modelpath, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        modelh5path = model_name+'/'+model_name+'.h5'
        model.save_weights(modelh5path)
        print("Saved model to disk")
    
    def loadModel(self,model_name):
        modelpath = model_name+'/'+model_name+'.json'
        json_file = open(modelpath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        modelh5path = model_name+'/'+model_name+'.h5'
        model.load_weights(modelh5path)
        print("Loaded model from disk")
        return model
    
    def trainModel(self, num_episodes):
        last_time = time.time()
        current_game_state = self.game.getGameScreen()
        current_state_stack = np.stack((current_game_state,current_game_state,current_game_state,current_game_state), axis=2) # stack 4 images to create placeholder input
        current_state_stack = current_state_stack.reshape(IMAGE_SIZE)  #1*84*84*4


        episode_duration_list = []
        episode_reward_list = []
        episode_loss_list = []


        for episode in range(num_episodes):
            print('Episode: {}...'.format(episode))
            # starttime for an episode
            episode_start = time.time()
            while(True):
                # one game episode/gameover
                gameOver_Status = self.game.gameOver()
                if gameOver_Status['status']:
                    episode_end = time.time()
                    episode_duration = episode_end-episode_start
                    episode_duration_list.append(episode_duration)
                    episode_reward_list.append(gameOver_Status['game_score'])
                    # check if enough traning samples are available, train the model
                    if self.replaymemory.can_sample(BATCH_SIZE):
                        print('Enough batch size available, starting training...')
                        current_stateList=[]
                        actionList=[]
                        next_stateList=[]
                        rewardList=[]

                        memory = self.replaymemory.sample(BATCH_SIZE)
                        # gets list of experience
                        # ('state', 'action', 'next_state', 'reward')
                        for m in memory:
                            current_stateList.append(m.state)
                            actionList.append(m.action)
                            next_stateList.append(m.next_state)
                            rewardList.append(m.reward)
                        
                        
                        current_stateList = np.array(current_stateList)
                        qvalue_current_state = self.model.predict(current_stateList)
                        next_stateList = np.array(next_stateList)
                        qvalue_next_state = self.target_model.predict(next_stateList)
                        x = []
                        y = []
                        # calculate target Qvalue
                        for index, (state, action, next_state, reward) in enumerate(memory):
                            max_future_q = np.max(qvalue_next_state[index])
                            new_q = reward + GAMMA * max_future_q
                            # Update Q value for given state
                            current_qs = qvalue_current_state[index]
                            current_qs[action] = new_q

                            # And append to our training data
                            x.append(state)
                            y.append(current_qs)

                        # fit the model
                        history = self.model.fit(np.array(x), np.array(y),batch_size=BATCH_SIZE, verbose=1, epochs=1)
                        episode_loss_list.append(history.history['loss'][0])
                    
                    
                    # check if we have to update the model weights
                    if episode % TARGET_UPDATE == 0:
                        print('Updating target_model weights...')
                        self.target_model.set_weights(self.model.get_weights())
                        self.saveModel('model',self.model)
                        self.saveModel('target_model',self.target_model)
                    
                    self.game.restartGame()
                    break;
                else:
                    # choose action based of exploration vs exploitation
                    rate = self.strategy.get_exploration_rate(self.current_step)
                    rand = random.random()
                    self.current_step += 1
                    if rate > rand:
                        # taking random action based on epsilon
                        action = random.randrange(NUM_ACTIONS)
                        print('***Random action***')
                        self.game.takeAction(action)
                    else:
                        print('***Qvalue action***')
                        qvalues = self.getTargetModelPredict(current_state_stack.reshape(STACK_IMAGE_SIZE))
                        print(qvalues)
                        action = np.argmax(qvalues[0])
                        self.game.takeAction(action)
                    self.reward = gameOver_Status['score']
                    print(self.reward)
                    next_state = self.game.getGameScreen()
                    print('fps: {0}'.format(1 / (time.time()-last_time))) # helpful for measuring frame rate
                    last_time = time.time()
                    next_state_stack = np.append(next_state, current_state_stack[:, :, :3])
                    next_state_stack = next_state_stack.reshape(IMAGE_SIZE) 
                    ex = self.experience(current_state_stack,action,next_state_stack,self.reward)
                    self.replaymemory.push(ex)
                    

            
        # plot final results
        self.plotEpisodeLoss(num_episodes,episode_loss_list)
        self.plotEpisodeDuration(num_episodes,episode_duration_list)
        self.plotEpisodeReward(num_episodes,episode_reward_list)

    
    def getModelPredict(self,state):
        # convert to a 4-dim (batchsize,height,width,chaneel)
        return self.model.predict(state) 

        
    def getTargetModelPredict(self,state):
        # convert to a 4-dim (batchsize,height,width,chaneel)
        return self.target_model.predict(state) 
        
    # model to test the DQN aka play game
    def playModel(self):
        current_game_state = self.game.getGameScreen()
        current_state_stack = np.stack((current_game_state,current_game_state,current_game_state,current_game_state), axis=2) # stack 4 images to create placeholder input
        current_state_stack = current_state_stack.reshape(STACK_IMAGE_SIZE)  #1*84*84*4
        
        while(True):
            gameOver_Status={
                'status':False,
                'score':0,
                'game_score':0
                }
            last_time = time.time()
            gameOver_Status = self.game.gameOver()
            if gameOver_Status['status']:
                self.game.restartGame()
            else:
                print('***Qvalue action***')
                qvalues = self.getTargetModelPredict(current_state_stack)
                print(qvalues)
                action = np.argmax(qvalues[0])
                self.game.takeAction(action)
                gameOver_Status = self.game.gameOver()
                self.reward = gameOver_Status['score']

                next_state = self.game.getGameScreen()
                print('fps: {0}'.format(1 / (time.time()-last_time))) # helpful for measuring frame rate
                last_time = time.time()
                next_state_stack = np.append(next_state, current_state_stack[:, :, :, :3])
                next_state_stack = next_state_stack.reshape(STACK_IMAGE_SIZE)

    #plot graph for loss/episodes
    def plotEpisodeLoss(self,num_episodes,episode_loss):
        # Plot training & validation accuracy values
        x = range(0, len(episode_loss))
        y = episode_loss
        plt.plot(x,y)
        plt.title('Model loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.savefig('plots/episodeloss.png', bbox_inches='tight')
        plt.close()           
    # plot graph for episode duration/episodes
    def plotEpisodeDuration(self,num_episodes,episode_duration):
        x = range(0, len(episode_duration))
        y = episode_duration
        plt.plot(x,y)
        plt.title('Survival over time')
        plt.xlabel('Episode')
        plt.ylabel('Episode duration')
        plt.savefig('plots/episodeduration.png', bbox_inches='tight')
        plt.close()

    # plot graph for episode duration/reward
    def plotEpisodeReward(self,num_episodes,episode_reward):
        x = range(0, len(episode_reward))
        y = episode_reward
        plt.plot(x,y)
        plt.title('Reward over time')
        plt.xlabel('Episode')   
        plt.ylabel('Episode reward')
        plt.savefig('plots/episodereward.png', bbox_inches='tight')
        plt.close()