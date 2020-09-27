"""
# @Time    :  2020/9/20
# @Author  :  Jimou Chen
"""
from itertools import permutations

s = list(input())
s.sort()
a = ''
for i in s:
    a += i

ll = []
per = permutations(a, len(a))
for i in per:
    l = list(i)
    out = ''
    for j in l:
        out += j
    ll.append(out)

for each in ll:
    if each != ll[-1]:
        print(each)
    else:
        print(each, end='')
