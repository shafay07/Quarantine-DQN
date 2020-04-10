import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
IMAGE_SIZE = (620, 450, 1)
ACTION_SPACE_SIZE = 3


class dqn:
    def __init__(self):
        print('initializing DQN Agent')
        self.reward = 0

        # main model,model for fitting
        self.model = self.createModel()
        print(self.model.summary())

        # target network, clone of main model for predictions
        self.target_model = self.createModel()
        self.target_model.set_weights(self.model.get_weights())

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

    def getQvalue(self, state):
        return self.target_model.predict(state)
