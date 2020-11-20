"""
# @Time    :  2020/11/20
# @Author  :  Jimou Chen
"""
from collections import Counter

while True:
    try:
        n = int(input())
        l = list(map(int, input().split()))
        res = dict(Counter(l))
        max_item = max(res.items(), key=lambda x: x[1])

        if max_item[1] * 2 > n:
            print(max_item[0])
        else:
            print('0')

    except:
        break

'''
4
1 2 2 2

7
14 2 14 14 3 14 6
'''
