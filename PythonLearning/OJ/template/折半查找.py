"""
# @Time    :  2020/10/21
# @Author  :  Jimou Chen
"""


def find_half(array: list, low, high, key):
    if low >= high:
        return -1

    else:
        mid = (low + high) // 2
        if array[mid] == key:
            return mid
        elif array[mid] > key:
            return find_half(array, low, mid - 1, key)
        else:
            return find_half(array, mid + 1, high, key)


if __name__ == '__main__':
    test = [1, 2, 3, 4, 5, 7, 8]
    res = find_half(test, 0, len(test), 17)
    print(res)
