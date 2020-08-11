"""
# @Time    :  2020/8/2
# @Author  :  Jimou Chen
"""
import threading
# 访问全局变量count，对其作修改
count = 1
# 设置锁对象
lock = threading.Lock()


def run_thread(n):
    # 获取锁
    lock.acquire()
    try:
        global count
        for i in range(500000):
            count += n
            count -= n
    finally:
        # 释放锁
        lock.release()


if __name__ == '__main__':
    th1 = threading.Thread(target=run_thread, args=(3,))
    th2 = threading.Thread(target=run_thread, args=(3,))

    th1.start()
    th2.start()

    th1.join()
    th2.join()

    print(count)
