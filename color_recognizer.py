import pyautogui
import time
from tkinter import *
import keyboard
import threading

class GUI:
    def __init__(self):
        self.__mainWindow = Tk()
        self.__mainWindow.title("Color Recognizer")
        self.__mainWindow.option_add("*Font", "Verdana 30")
        self.__mainWindow.mainloop()

def getPixel():
    previousPixel = None
    while True:
        if keyboard.is_pressed("a"):
            pos = pyautogui.position()
            if previousPixel == pos:
                continue
            previousPixel = pos
            im = pyautogui.screenshot()
            print(im.getpixel((pos)))
            time.sleep(1)

def main():
    t1 = threading.Thread(target=getPixel)
    t1.start()
    GUI()
    

if __name__ == "__main__":
    main()