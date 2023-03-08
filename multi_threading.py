from threading import Thread
import numpy as np


class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def func(n1, n2):
    return np.array([[n1, n2], [n1, n2]]), np.array([[n1, n2], [n1, n2], [n1, n2]])


if __name__ == '__main__':
    thread_list = []
    for i in range(3):
        thread = CustomThread(target=func, args=(5, 3,))
        thread_list.append(thread)
    for i in thread_list:
        i.start()
    for i in thread_list:
        a = i.join()
        print(a[1])
        # print(b)
        print()
