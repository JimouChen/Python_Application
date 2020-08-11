"""
# @Time    :  2020/8/2
# @Author  :  Jimou Chen
"""
import time
import multiprocessing


def work():
    for i in range(10):
        print('I am working...')
        time.sleep(0.2)


if __name__ == '__main__':
    work_process = multiprocessing.Process(target=work)
    # 设置守护进程
    work_process.daemon = True  # 加上这句的话，父进程一结束，子进程就结束
    work_process.start()
    # work_process.join()  # 加上这句的话，父进程会等子进程结束后再运行
    time.sleep(1)
    print('main process is over')
