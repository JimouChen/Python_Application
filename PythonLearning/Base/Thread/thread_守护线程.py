"""
# @Time    :  2020/8/2
# @Author  :  Jimou Chen
"""
import threading
from time import sleep


def work():
    for i in range(10):
        print('working ...')
        sleep(0.2)


if __name__ == '__main__':
    sub_thread = threading.Thread(target=work)
    # sub_thread = threading.Thread(target=work, daemon=True) # daemon=True是设置了守护线程
    sub_thread.setDaemon(True)
    sub_thread.start()
    # sub_thread.join()
    sleep(1)
    print('main thread is over')
