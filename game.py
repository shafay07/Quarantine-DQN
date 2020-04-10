import numpy as np
import pytesseract as pt
from PIL import ImageGrab
import cv2
import pyautogui
import sys
import time
from dqn import dqn


# gameScreen.reshape(number of images(4 for stack), height(620), width(450), dimension(1))
class game:
    def __init__(self):
        # path of tesseract executable
        pt.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'
        print('Game world initialized.')
        self.agent = dqn()

        # starting pos for catcher (480,795)
        pyautogui.moveTo(480, 795)

    def trainGame(self, epoch):
        # windowed mode for the game env, at the left position of your main screen.
        # convert is to get grayscle image
        statelist = []
        while(epoch):
            gameScreen = np.array(ImageGrab.grab(
                bbox=(250, 250, 700, 870)).convert('L'))

            statelist.append(gameScreen)
            cv2.imshow('window', gameScreen)
            epoch -= 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            # one game episode/gameover
            if self.gameOver(gameScreen):
                self.restartGame()
            print('****GETTING QVALUE****')
            gameScreen = gameScreen.reshape(1, 620, 450, 1)
            print(self.agent.getQvalue(gameScreen))

    def playGame(self):
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

    def restartGame(self):
        print('Restarting the game..')
        time.sleep(2)
        pyautogui.click(x=92, y=65)
        time.sleep(2)
        pyautogui.click(x=475, y=565)
        time.sleep(2)
        pyautogui.moveTo(480, 795)

    def gameOver(self, screen):
        screen = screen[:200, :400]
        gameOverStr = pt.image_to_string(screen, lang='eng')
        # while(True):
        #     cv2.imshow('window', screen)
        #     if cv2.waitKey(25) & 0xFF == ord('q'):
        #         cv2.destroyAllWindows()
        #         break

        if gameOverStr == 'GAMEOVER':
            return True
        else:
            return False

    def noDrag(self):
        print('taking no action')
        return None

    def leftDrag(self):
        print('dragging left')
        return pyautogui.dragRel(-100, 0, duration=0.5)  # drag left

    def rightDrag(self):
        print('dragging right')
        return pyautogui.dragRel(100, 0, duration=0.5)  # drag right
