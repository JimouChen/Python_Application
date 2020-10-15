"""
# @Time    :  2020/10/15
# @Author  :  Jimou Chen
"""


def bin_search(a: list, low, high, key):
    if low > high:
        return 0
    mid = (low + high) // 2
    if a[mid] == key:
        return mid
    if a[mid] > key:
        return bin_search(a, low, mid - 1, key)
    else:
        return bin_search(a, mid + 1, high, key)


if __name__ == '__main__':
    test = [2, 3, 4, 6, 7, 9]
    res = bin_search(test, 0, len(test), 7)
    print(res)
