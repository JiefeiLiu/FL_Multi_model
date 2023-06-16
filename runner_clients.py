import time
import subprocess
import os

from pynput.keyboard import Key, Controller

keyboard = Controller()


def create_screen(com, activate_conda=False):
    keyboard.type(com)
    time.sleep(0.5)
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    if activate_conda:
        time.sleep(0.5)
        keyboard.type("conda activate pytorch_cuda")
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
        time.sleep(0.5)
    time.sleep(1)
    exit_screen()


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
    act_conda = False
    create_scn = False
    run_client = True
    if create_scn:
        for i in range(1, num_client+1):
            create_screen_name = "screen -S client" + str(i)
            create_screen(create_screen_name, activate_conda=act_conda)
            time.sleep(1)

    if run_client:
        for i in range(1, num_client+1):
            screen_name = "screen -r client" + str(i)
            enter_screen(screen_name)
            time.sleep(1)
            execute_program(i-1)
            time.sleep(2)
            exit_screen()
            time.sleep(1)

