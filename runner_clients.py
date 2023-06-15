import time
import subprocess
import os

from pynput.keyboard import Key, Controller

keyboard = Controller()


def enter_screen(com):
    keyboard.type(com)
    time.sleep(0.3)
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)


def execute_program(index):
    keyboard.type('python client.py ' + str(index))
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)


def exit_screen():
    keyboard.press(Key.ctrl)
    keyboard.press('a')
    keyboard.press('d')

    keyboard.release(Key.ctrl)
    keyboard.release('a')
    keyboard.release('d')


    # keyboard.press(Key.ctrl)
    # keyboard.press('c')
    # keyboard.release(Key.ctrl)
    # keyboard.release('c')


if __name__ == '__main__':
    time.sleep(3)
    num_client = 20
    for i in range(1, num_client+1):
        screen_name = "screen -r client" + str(i)
        enter_screen(screen_name)
        time.sleep(1)
        execute_program(i-1)
        time.sleep(2)
        exit_screen()
        time.sleep(1)

