"""
# @Time    :  2020/8/1
# @Author  :  Jimou Chen
"""
import threading
from time import sleep


# 继承线程类
class MyThread(threading.Thread):
    def __init__(self, thread_name):
        super().__init__()
        self.thread_name = thread_name

    def run(self):
        print('thread start:' + self.getName() + '\t' + self.thread_name)
        count = 5
        while count:
            sleep(1)
            print(self.getName() + ' : count = %d' % count)
            count -= 1

        print('thread over...:' + self.getName())


thread_1 = MyThread('线程1')
thread_2 = MyThread('线程2')
thread_1.start()
thread_2.start()
thread_1.join()
thread_2.join()
