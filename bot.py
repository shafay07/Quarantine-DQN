import time
import pyautogui
import cv2
import numpy as np
from PIL import ImageGrab
import cv2
from game import game
from dqn import dqn
from mss import mss

# Constant hyper-parameters
NUM_EPISODES = 1001
TRAIN = True

# helper funtion to get screen-cordinates, call it to get real time cordinates


def getScreenCordinates():
    try:
        while True:
            x, y = pyautogui.position()
            positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
            print(positionStr, end='')
            print('\b' * len(positionStr), end='', flush=True)
    except KeyboardInterrupt:
        print('\n')


def main():
    
    
    # getScreenCordinates()
    agent = dqn()
    if TRAIN:
        agent.trainModel(NUM_EPISODES)
    else:
        agent.playModel()


if __name__ == "__main__":
    main()
