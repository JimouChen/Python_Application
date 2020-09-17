"""
# @Time    :  2020/9/17
# @Author  :  Jimou Chen
"""



def T(n):
    if n == 1:
        return 4
    elif n > 1:
        return 3 * T(n - 1)


def T(n):
    if n == 1:
        return 1
    elif n > 1:
        return 2 * T(n // 3) + n


print(T(5))
