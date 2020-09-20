"""
# @Time    :  2020/9/20
# @Author  :  Jimou Chen
"""
from itertools import permutations

s = input()
per = permutations(s, len(s))
for i in per:
    l = list(i)
    out = ''
    for j in l:
        out += j
    print(out)

