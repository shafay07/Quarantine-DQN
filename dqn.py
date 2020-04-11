import numpy as np
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
IMAGE_SIZE = (620, 450, 1)
ACTION_SPACE_SIZE = 3
BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 0.001
TARGET_UPDATE = 10 # update target_network every 10 episodes
MEMORY_SIZE = 100000
LEARNING_RATE = 0.001

# one episode is one complete gameplay from start till end state(gameover state)
class dqn:
    def __init__(self):
        print('Initializing DQN Agent')
        self.current_step = 0
        
        self.replaymemory = replaymemory(MEMORY_SIZE)
        self.strategy = EpsilonGreedyStrategy(EPS_START,EPS_END,EPS_DECAY)
        self.game = game()
        # main model,model for fitting
        self.model = self.createModel()
        # target network, clone of main model for predictions
        self.target_model = self.loadModel('target_model')
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
        model.add(Dense(ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(
            lr=0.001), metrics=['accuracy'])
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
        target_model = model_from_json(loaded_model_json)
        # load weights into new model
        modelh5path = model_name+'/'+model_name+'.h5'
        target_model.load_weights(modelh5path)
        print("Loaded model from disk")
        return target_model
    
    def trainModel(self, num_episodes):
        # windowed mode for the game env, at the left position of your main screen.
        # convert is to get grayscle image
        episode_duration = []
        for episode in range(num_episodes):
            # restart game for every episode
            self.game.restartGame
            self.reward = 0
            while(True):
                # grabs the game screen
                gameScreen = np.array(ImageGrab.grab(
                    bbox=(250, 250, 700, 870)).convert('L'))
                print(type(gameScreen))
                cv2.imshow('window', gameScreen)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

                # choose action based of exploration vs exploitation
                rate = self.strategy.get_exploration_rate(self.current_step)
                self.current_step += 1
                if rate > random.random():
                    # taking random action based on epsilon
                    action = random.randrange(NUM_ACTIONS)
                    print('***Taking random action***')
                    self.game.takeAction(action)
                else:
                    print('****GETTING QVALUE****')
                    gameScreen = gameScreen.reshape(1, 620, 450, 1)
                    qvalues = self.getQvalue(gameScreen)
                    action = np.argmax(qvalues[0])
                    self.game.takeAction(action)

                # one game episode/gameover
                if self.game.gameOver(gameScreen):
                    self.reward = 0
                    self.game.restartGame()
                    break;
                else:
                    self.reward += 1
                    print(self.reward)
                    next_state = np.array(ImageGrab.grab(
                    bbox=(250, 250, 700, 870)).convert('L'))
                    next_state = next_state.reshape(1, 620, 450, 1)
                    e = self.experience(gameScreen,action,next_state,self.reward)
                    self.replaymemory.push(self.experience)
                    print(e)

                
            
    def playModel(self):
        statelist = []
        gameScreen = np.array(ImageGrab.grab(
            bbox=(250, 250, 700, 870)).convert('L'))
        while(not self.gameOver(gameScreen)):
            gameScreen = np.array(ImageGrab.grab(
                bbox=(250, 250, 700, 870)).convert('L'))
            statelist.append(gameScreen)
            cv2.imshow('window', gameScreen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            

    # estimate q-value given a state
    def getQvalue(self, state):
        return self.target_model.predict(state)
