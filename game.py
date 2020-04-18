import numpy as np
import pytesseract as pt
from PIL import ImageGrab, Image
import cv2
import pyautogui
import sys
import time

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


# gameScreen.reshape(number of images(4 for stack), height(620), width(450), dimension(1))
class game:
    def __init__(self):
        pyautogui.FAILSAFE = False
        self.lastScore = 0
        self.stateScore = 0
        self.initSession()
        
        print('Game world initialized.')
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
            'score':0
        }
        if logs:
            record = logs[-1]
            score_str = record.get("message").split('\"')[1]
            score_int = score_str.split(': ')[1]
            if len(score_int) <= 3:
                score_int = int(score_int)

            if score_int == self.lastScore:
                print('Didnt catch...')
                self.stateScore-=1
                context['score']=self.stateScore
            else:
                print('Catch...')
                self.stateScore+=2
                context['score']=self.stateScore
            
            self.lastScore = score_int
            context['status'] = False

            return context
        elif not logs:
            print('Game over...')
            self.driver.quit()  
            context['status'] = True
            return context
            

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
            
