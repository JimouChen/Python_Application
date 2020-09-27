"""
# @Time    :  2020/9/27
# @Author  :  Jimou Chen
"""

while True:
    try:
        n = int(input())
        l = list(map(int, input().split(' ')))
        flag = [0 for i in range(0, n)]
        for i in range(n):
            for j in range(i + 1, n):
                if l[i] == l[j]:
                    flag[i] = 1
                    flag[j] = 1
                    break

        ss = 0
        for i in range(n):
            if flag[i] == 0:
                ss += l[i]
        print(ss)
    except:
        break
