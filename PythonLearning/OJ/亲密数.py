"""
# @Time    :  2020/6/4
# @Author  :  Jimou Chen
"""


def fac(n):
    for a in range(2, n + 1):
        b = 0
        for i in range(1, a):
            if a % i == 0:
                b += i
        d = 0
        for i in range(1, b):
            if b % i == 0:
                d += i
        if d == a and d != b:
            if a < b:
                print("{}-{}".format(a, b))


n = int(input())
e = fac(n)
