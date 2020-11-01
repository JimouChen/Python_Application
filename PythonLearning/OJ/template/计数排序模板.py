"""
# @Time    :  2020/10/29
# @Author  :  Jimou Chen
"""


def count_sort(array: list):
    max_val = max(array)
    cnt = [0 for _ in range(max_val + 1)]

    for i in array:
        cnt[i] += 1

    print(cnt)
    for i in range(len(cnt)):
        print(str(i) * cnt[i], end='')


if __name__ == '__main__':
    test = list(map(int, input().split()))
    count_sort(test)
'''
2 3 4 1 2 2 4
'''