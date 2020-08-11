"""
# @Time    :  2020/7/18
# @Author  :  Jimou Chen
"""


class A(str):
    def __new__(cls, string):
        string = string.upper()
        return str.__new__(cls, string)


a = A('hello World')
print(a)


class B:
    def __init__(self):
        print('构造函数被调用')

    def __del__(self):
        print('析构函数被调用')


b1 = B()
b2 = b1
b3 = b2
del b3
del b2
del b1


# 重写加减法,继承int类
class Int(int):
    def __add__(self, other):
        return int.__sub__(self, other)

    def __sub__(self, other):
        return int(self) + int(other)
        # 或者return int.__add__(self, other)


a = Int(5)
b = Int(4)
print(a + b, a - b)
