"""
# @Time    :  2020/10/2
# @Author  :  Jimou Chen
"""
t = int(input())

while t:
    f = [0 for i in range(0, 50)]
    t -= 1
    n, m = map(int, input().split())
    f[0] = 1
    for i in range(1, n + 1):
        if i < m:
            f[i] = 2 * f[i - 1]
        if i == m:
            f[i] = 2 * f[i - 1] - 1
        if i > m:
            f[i] = 2 * f[i - 1] - f[i - m - 1]

    print(f[n])
