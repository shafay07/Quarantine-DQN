import numpy as np
import pytesseract as pt
from PIL import ImageGrab, Image
import cv2
import pyautogui
import sys
import time



# gameScreen.reshape(number of images(4 for stack), height(620), width(450), dimension(1))
class game:
    def __init__(self):
        # path of tesseract executable
        pt.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'
        print('Game world initialized.')
        # starting pos for catcher (480,795)
        self.spritepos = (480, 795)
        pyautogui.moveTo(self.spritepos)

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
        i = Image.fromarray(screen,'L')
        i = np.array(i)
        print(type(i))
        print(i)
        gameOverStr = pt.image_to_string(i, lang='eng')
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
            
