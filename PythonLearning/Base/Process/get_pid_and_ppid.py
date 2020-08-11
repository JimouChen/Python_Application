"""
# @Time    :  2020/8/2
# @Author  :  Jimou Chen
"""
import time
import multiprocessing
import os


# 多进程
def sing(num, name):
    print('pid of sing = ', os.getpid())
    print('ppid of sing = ', os.getppid())

    for i in range(num):
        print(name + ' is singing ...')
        time.sleep(1)


def dance(num, name):
    print('pid of dance = ', os.getpid())
    print('ppid of dance = ', os.getppid())

    for i in range(num):
        print(name + ' is dancing ...')
        time.sleep(1)


if __name__ == '__main__':
    print('pid of main process = ', os.getpid())
    # 使用进程类创建进程对象
    sing_process = multiprocessing.Process(target=sing, args=(3, 'AA'))
    dance_process = multiprocessing.Process(target=dance, kwargs={'name': 'BB', 'num': 2})

    # 启动进程
    sing_process.start()
    dance_process.start()

    sing_process.join()
    dance_process.join()
