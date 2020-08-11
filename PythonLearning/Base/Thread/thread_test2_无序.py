"""
# @Time    :  2020/8/2
# @Author  :  Jimou Chen
"""
import threading
from time import sleep

'''验证子线程的无序'''


def show_msg():
    sleep(1)
    # 获取当前的线程对象
    thread = threading.current_thread()
    print(thread, 'id = ', threading.get_ident())


if __name__ == '__main__':
    # thread_list = []
    # 创建5个线程
    for i in range(5):

        sub_thread = threading.Thread(target=show_msg, name='thread'+str(i))
        # thread_list.append(sub_thread)
        sub_thread.start()
    #
    # for i in range(5):
    #     thread_list[i].join()