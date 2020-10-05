"""
# @Time    :  2020/10/2
# @Author  :  Jimou Chen
"""
def is_win(a: list, b: list, n):
    a.sort()
    b.sort()
    i = n - 1
    j = n // 2
    while i >= n // 2:
        if a[i] <= b[j]:
            return 0
        i -= 1
        j -= 1

    return 1


while True:
    try:
        n = int(input())
        a = list(map(int, input().split()))
        b = list(map(int, input().split()))
        if is_win(a, b, n) == 1:
            print('YES')
        else:
            print('NO')

    except:
        break
