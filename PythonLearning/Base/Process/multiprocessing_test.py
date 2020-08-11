"""
# @Time    :  2020/8/2
# @Author  :  Jimou Chen
"""
import time
import multiprocessing


# 多进程
def sing():
    for i in range(3):
        print('singing ...')
        time.sleep(1)


def dance():
    for i in range(3):
        print('dancing ...')
        time.sleep(1)


if __name__ == '__main__':
    # 使用进程类创建进程对象
    sing_process = multiprocessing.Process(target=sing)
    dance_process = multiprocessing.Process(target=dance)

    # 启动进程
    sing_process.start()
    dance_process.start()

    sing_process.join()
    dance_process.join()