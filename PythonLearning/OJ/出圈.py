"""
# @Time    :  2020/7/15
# @Author  :  Jimou Chen
"""

'''
也可以公式法：f(n, m) = (f(n - 1) + m) mod n
'''


def josef(n, m):
    number = []

    for i in range(n):
        number.append(i + 1)

    remove_index = 0
    while len(number) != 1:
        remove_index = (remove_index + m - 1) % len(number)
        number.remove(number[remove_index])

    return number[0]


while True:
    n, m = map(int, input().split())
    print(josef(n, m))
