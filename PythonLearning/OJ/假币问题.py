"""
# @Time    :  2020/9/20
# @Author  :  Jimou Chen
"""
import math

while True:
    try:
        n = int(input())
        if n == 0:
            break
        res = math.ceil(math.log(n) / math.log(3))
        print(res)
    except:
        break
