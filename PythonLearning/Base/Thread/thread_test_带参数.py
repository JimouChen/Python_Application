"""
# @Time    :  2020/8/2
# @Author  :  Jimou Chen
"""
import time
import threading


def sing(num):
    for i in range(num):
        print('singing ...')
        time.sleep(1)


def dance(count):
    for i in range(count):
        print('dancing ...')
        time.sleep(1)


if __name__ == '__main__':
    # 创建线程
    sing_thread = threading.Thread(target=sing, args=(3,))
    dance_thread = threading.Thread(target=dance, kwargs={'count': 4})

    # 启动线程
    sing_thread.start()
    dance_thread.start()

    sing_thread.join()
    dance_thread.join()
