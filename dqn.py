import numpy as np
import matplotlib.pyplot as plt
import math
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam,SGD
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
IMAGE_SIZE = (600, 440, 1)
ACTION_SPACE_SIZE = 3
BATCH_SIZE = 4
GAMMA = 0.9
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 0.001
TARGET_UPDATE = 5
MEMORY_SIZE = 10000
LEARNING_RATE = 0.001

# one episode is one complete gameplay from start till end state(gameover state)
class dqn:
    def __init__(self):
        print('Initializing DQN Agent')
        self.current_step = 0
        self.reward = 0
        self.replaymemory = replaymemory(MEMORY_SIZE)
        self.strategy = EpsilonGreedyStrategy(EPS_START,EPS_END,EPS_DECAY)
        self.game = game()
        # main model,model for fitting
        self.model = self.createModel()
        self.model.summary()
        # target network, clone of main model for predictions
        self.target_model = self.createModel()
        self.target_model.set_weights(self.model.get_weights())
        
        
        # self.model = self.loadModel('model')
        # self.target_model = self.loadModel('target_model')
        self.experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

    def createModel(self):
        model = Sequential()

        model.add(Convolution2D(32, (3, 3), input_shape=IMAGE_SIZE))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Convolution2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # this converts our 3D feature maps to 1D feature vectors
        model.add(Flatten())
        model.add(Dense(64))

        # ACTION_SPACE_SIZE = how many choices (3)
        model.add(Dense(ACTION_SPACE_SIZE, activation='softmax'))
        model.compile(loss="mean_squared_error", optimizer=SGD(
            lr=LEARNING_RATE))
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
        # windowed mode for the game env, at the left position of your main screen.
        # convert is to get grayscle image
        episode_duration_list = []
        episode_reward_list = []
        for episode in range(num_episodes):
            print('Episode: {}...'.format(episode))
            # starttime for an episode
            episode_start = time.time()
            while(True):
                # grabs the game screen
                gameScreen = np.array(ImageGrab.grab(
                    bbox=(260, 250, 700, 850)).convert('L'))
                current_state = gameScreen.reshape(1, 600, 440, 1)

                # one game episode/gameover
                gameOver_Status = self.game.gameOver()
                if gameOver_Status['status']:
                    episode_end = time.time()
                    episode_duration = episode_end-episode_start
                    episode_duration_list.append(episode_duration)
                    episode_reward_list.append(self.reward)
                    # check if enough traning samples are available, train the model
                    if self.replaymemory.can_sample(BATCH_SIZE):
                        print('Enough batch size available, starting training...')
                        current_stateList=[]
                        actionList=[]
                        next_stateList=[]
                        rewardList=[]

                        current_qs_list=[]
                        future_qs_list=[]
                        target_qs_list=[]

                        memory = self.replaymemory.sample(BATCH_SIZE)
                        # gets list of experience
                        for m in memory:
                            current_stateList.append(m.state)
                            actionList.append(m.action)
                            next_stateList.append(m.next_state)
                            current_state_reward = m.reward
                            rewardList.append(current_state_reward)
                            qvalue_current_state = np.max(self.model.predict(m.state))
                            current_qs_list.append(qvalue_current_state)
                            qvalue_next_state = np.max(self.target_model.predict(m.state))
                            future_qs_list.append(qvalue_next_state)
                            qvalue_target = current_state_reward + (GAMMA * qvalue_next_state)
                            target_qs_list.append(qvalue_target)
                        # fit the model
                        loss_list = []
                        
                        for i in range(len(current_stateList)):
                            print('------')
                            print(current_qs_list[i])
                            print('------')
                            print(target_qs_list[i])
                            history = self.model.fit(current_stateList[i],target_qs_list[i],batch_size=BATCH_SIZE,verbose=1, epochs=1)
                            loss_list.append(history.history['loss'])    

                        
                        
                        
                        

                        if episode % TARGET_UPDATE == 0:
                            print('Saving loss plot...')
                            # Plot training & validation accuracy values
                            plt.plot(range(BATCH_SIZE),loss_list)
                            plt.title('Model loss')
                            plt.ylabel('Loss')
                            plt.xlabel('Btach')
                            plot_dir = 'plots/'+str(episode)+'-loss-batch.png'
                            plt.savefig(plot_dir, bbox_inches='tight')
                            plt.close()


                    # check if we have to update the model weights
                    if episode % TARGET_UPDATE == 0:
                        print('Updating target_model weights...')
                        self.target_model.set_weights(self.model.get_weights())
                        self.saveModel('model',self.target_model)
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
                        qvalues = self.getQvalue(current_state)
                        print(qvalues)
                        action = np.argmax(qvalues[0])
                        self.game.takeAction(action)

                    self.reward = gameOver_Status['score']
                    next_state = np.array(ImageGrab.grab(
                    bbox=(260, 250, 700, 850)).convert('L'))
                    next_state = next_state.reshape(1, 600, 440, 1)
                    ex = self.experience(current_state,action,next_state,self.reward)
                    self.replaymemory.push(ex)
                    

            

        # plot final results
        self.plotEpisodeDuration(num_episodes,episode_duration_list)
        self.plotEpisodeReward(num_episodes,episode_reward_list)

    # model to test the DQN aka play game
    def playModel(self):
        
        while(True):
            gameOver_Status = self.game.gameOver()
            if gameOver_Status['status']:
                self.game.restartGame()
            else:
                gameScreen = np.array(ImageGrab.grab(
                    bbox=(260, 250, 700, 850)).convert('L'))
                current_state = gameScreen.reshape(1, 600, 440, 1)
                print('***Taking action***')
                qvalues = self.getQvalue(current_state)
                action = np.argmax(qvalues[0])
                self.game.takeAction(action)
            

            

    # estimate q-value given a state
    def getQvalue(self, state):
        return self.target_model.predict(state)

    # plot graph for episode duration/episodes
    def plotEpisodeDuration(self,num_episodes,episode_duration):
        x = range(0, num_episodes)
        y = episode_duration
        plt.plot(x,y)
        plt.title('Survival over time')
        plt.xlabel('Episode')
        plt.ylabel('Episode duration')
        plt.savefig('plots/episodeduration.png', bbox_inches='tight')
        plt.close()


    # plot graph for episode duration/reward
    def plotEpisodeReward(self,num_episodes,episode_reward):
        x = range(0, num_episodes)
        y = episode_reward
        plt.plot(x,y)
        plt.title('Reward over time')
        plt.xlabel('Episode')   
        plt.ylabel('Episode reward')
        plt.savefig('plots/episodereward.png', bbox_inches='tight')
        plt.close()