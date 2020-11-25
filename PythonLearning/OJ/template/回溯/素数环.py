"""
# @Time    :  2020/11/25
# @Author  :  Jimou Chen
"""
from math import sqrt

n = int(input())
num = [0 for _ in range(n + 1)]
flag = [0 for _ in range(n + 1)]


# 判断素数
def prime(x):
    for i in range(2, int(sqrt(x)) + 1):
        if x % i == 0:
            return 0

    return 1


# x 是当前的数，v是满足条件的前一个数
def dfs(x, v):
    if x == n + 1:
        # 判断最后一个数和第一个数之和
        if prime(v + 1):
            for i in range(1, n + 1):
                print(num[i], end=' ')
            print()
            return  # return的位置是和for同一级的

    for i in range(1, n + 1):
        if flag[i] == 0 and prime(i + v):
            flag[i] = 1
            num[x] = i
            dfs(x + 1, i)
            flag[i] = 0


num[1] = 1
flag[1] = 1
dfs(2, 1)

'''
6
1 4 3 2 5 6 
1 6 5 2 3 4 


8
1 2 3 8 5 6 7 4 
1 2 5 8 3 4 7 6 
1 4 7 6 5 8 3 2 
1 6 7 4 3 8 5 2 
'''

'''
优化的话，可以使用素数表，这样就不用每次都遍历判断了

k = 0
# 素数表，1表示素数
def prime_table(x):
    global k
    l = [1 for _ in range(x + 1)]
    for i in range(2, x + 1):
        for k in range(2, int(sqrt(i)) + 1):
            if i % k == 0:
                l[i] = 0

    return l

prime = prime_table(n+100)

'''
