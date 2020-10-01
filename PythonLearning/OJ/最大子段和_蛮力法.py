"""
# @Time    :  2020/10/1
# @Author  :  Jimou Chen
"""
ll = [1, -2, 3, -4, 5, 6, -7, 4, 3, -3, 1]
# ll = [-2, 11, -4, 13, -5, -2]

all_max = []
for i in range(0, len(ll)):
    s1 = ll[i]
    s_max = 0
    for j in range(i + 1, len(ll)):
        s1 += ll[j]
        if s1 > s_max:
            s_max = s1

    all_max.append(s_max)

print(max(all_max))
