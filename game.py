import numpy as np
import pytesseract as pt
import matplotlib.pyplot as plt
from PIL import ImageGrab, Image
import cv2
import pyautogui
import sys
import time
from mss import mss
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

IMG_SIZE = (84,84)
# gameScreen.reshape(number of images(4 for stack), height(84), width(84), dimension(1))
class game:
    def __init__(self):
        print('Initialize Game World')
        pyautogui.FAILSAFE = False
        self.lastScore = 0
        self.stateScore = 0
        self.game_score = 0
        self.initSession()
    
        # starting pos for catcher (480,795)
        self.spritepos = (480, 795)
        pyautogui.moveTo(self.spritepos)

    def initSession(self):
        self.lastScore = 0
        self.stateScore = 0
        self.d = DesiredCapabilities.CHROME
        self.d['loggingPrefs'] = { 'browser':'ALL' }
        self.driver = webdriver.Chrome('D:/chromedriver.exe',desired_capabilities=self.d)
        # load the desired webpage
        self.driver.get('https://quarantine-play.firebaseapp.com/')
        
        time.sleep(2)
        pyautogui.click(x=475, y=565)
        time.sleep(1)
        self.spritepos = (480, 795)
        pyautogui.moveTo(self.spritepos)

    def restartGame(self):
        print('Restarting the game..')
        self.initSession()

    def gameOver(self):
        # using selenium, enable browser logging
        logs = self.driver.get_log('browser')
        context={
            'status':False,
            'score':0,
            'game_score':0
        }
        if logs:
            record = logs[-1]
            score_str = record.get("message").split('\"')[1]
            print(score_str)
            score_int = score_str.split(': ')[1]
            print(score_int)
            if len(score_int) <= 3:
                score_int = int(score_int)
                self.game_score = score_int
                

            if score_int == self.lastScore:
                print('Didnt catch...')
                self.stateScore-=1
                context['score']=-1
            else:
                print('Catch...')
                self.stateScore+=2
                context['score']=0.1
            
            self.lastScore = score_int
            context['status'] = False

            return context
        elif not logs:
            print('Game over...')
            self.driver.quit()  
            context['status'] = True
            context['game_score']=self.game_score
            return context
            
    def getGameScreen(self):
        print('Getting game state...')
        with mss() as sct:
            # part of the screen to capture
            bbox=(260, 250, 700, 850)
            # get raw pixels from the screen, save it to a Numpy array
            capture = np.array(sct.grab(bbox))
            # convert to Grayscale
            capture = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
            # resize into 84x84
            capture = cv2.resize(capture, IMG_SIZE)
        return capture
    def noDrag(self):
        print('taking no action')
        self.spritepos = pyautogui.position()

    def leftDrag(self):
        print('dragging left')
        pyautogui.moveTo(self.spritepos)
        pyautogui.dragRel(-100, 0, duration=0.5)  # drag left
        self.spritepos = pyautogui.position()
        

    def rightDrag(self):
        print('dragging right')
        pyautogui.moveTo(self.spritepos)
        pyautogui.dragRel(100, 0, duration=0.5)  # drag right
        self.spritepos = pyautogui.position()

    def takeAction(self,index):
        if index == 0:
            return self.leftDrag()
        elif index == 1:
            return self.rightDrag()
        elif index == 2:
            return self.noDrag()
        else:
            return None
            
