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
        keyboard.type("conda activate FL_env")
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


def execute_program(index, process_count, cuda_index, num_process_per_gpu):
    # print(process_count, cuda_index)
    if process_count < num_process_per_gpu:
        cuda_name = "cuda:" + str(cuda_index)
        com = 'python client.py ' + str(index) + ' ' + cuda_name
        process_count += 1
    else:
        cuda_index += 1
        process_count = 1
        cuda_name = "cuda:" + str(cuda_index)
        com = 'python client.py ' + str(index) + ' ' + cuda_name
    keyboard.type(com)
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    return process_count, cuda_index


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
    num_client = 50
    num_gpus = 4
    act_conda = False
    create_scn = False
    run_client = True
    if create_scn:
        client_lower_bound = 0
        for i in range(client_lower_bound+1, num_client+1):
            create_screen_name = "screen -S client" + str(i)
            create_screen(create_screen_name, activate_conda=act_conda)
            time.sleep(1)

    if run_client:
        num_process_per_gpu = num_client/num_gpus
        process_count = 0
        cuda_index = 0
        for i in range(1, num_client+1):
            screen_name = "screen -r client" + str(i)
            enter_screen(screen_name)
            time.sleep(1)
            process_count, cuda_index = execute_program(i-1, process_count, cuda_index, num_process_per_gpu)
            time.sleep(2)
            exit_screen()
            time.sleep(1)

