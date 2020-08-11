"""
# @Time    :  2020/7/18
# @Author  :  Jimou Chen
"""


def my_yield():
    print('生成器被调用')
    yield 1  # 第一次调用执行到这里结束，第二调用继续从下一句开始执行
    yield 2


my_gen = my_yield()  # 输出 生成器被调用
print(next(my_gen))  # 输出1
print(next(my_gen))  # 输出2


for i in my_yield():
    print(i)


# -------------------------------------------------------

def fab(n):
    a = 0
    b = 1
    while True:
        if a < n:
            a, b = b, a + b
            yield a
        else:
            break


for each in fab(100):
    print(each)
