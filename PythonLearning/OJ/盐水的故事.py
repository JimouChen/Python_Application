"""
# @Time    :  2020/9/27
# @Author  :  Jimou Chen
"""

t = 0
c = 0
i = 1
value, d = map(int, input().split())
while value:
    value -= d
    t += 1
    c += 1
    if c == i and value > 0:
        t += 1
        i += 1
        c = 0

    # print(t)
print(t)