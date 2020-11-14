"""
# @Time    :  2020/7/17
# @Author  :  Jimou Chen
"""


def hanoi(n, x, y, z):
    if n == 1:
        print(x, '-->', z)
    else:
        hanoi(n - 1, x, z, y)  # 将前n-1个盘子从x借助z移到y
        print(x, '-->', z)  # 再把最后一个(第n个)从x直接移到z
        hanoi(n - 1, y, x, z)  # 将剩下的n-1个盘子从y借助x移到z


n = int(input())
hanoi(n, 'a', 'b', 'c')
