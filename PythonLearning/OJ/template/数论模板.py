"""
# @Time    :  2020/11/8
# @Author  :  Jimou Chen
"""

import math


# gcd直接使用math里面的gcd

# 最大公约数
def lcm(a, b):
    return a * b // math.gcd(a, b)
