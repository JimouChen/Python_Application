"""
# @Time    :  2020/9/20
# @Author  :  Jimou Chen
"""
import math

a, b = map(int, input().split())
c, d = map(int, input().split())

res = complex(a, b) * complex(c, d)
print(int(res.real), int(res.imag))
