"""
# @Time    :  2020/7/17
# @Author  :  Jimou Chen
"""
'''
对这些小问题，用迭代解决会快一些
以下只是练习
'''


# n的阶乘
def fact(n):
    if n == 1 or n == 0:
        return 1
    else:
        return fact(n - 1) * n


# 斐波那契
def fab(n):
    if n < 1:
        print('输入有误')
        return -1
    elif n == 1 or n == 2:
        return 1
    else:
        return fab(n - 1) + fab(n - 2)


print(fact(3))
print(fab(6))
