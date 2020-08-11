"""
# @Time    :  2020/8/11
# @Author  :  Jimou Chen
"""


def bin_search(arr, key):
    low = 0
    high = len(arr) - 1
    found = False

    while low <= high and not found:
        mid = (low + high) // 2
        if arr[mid] == key:
            found = True
        else:
            if arr[mid] < key:
                low = mid + 1
            else:
                high = mid - 1

    return found


testList = [1, 2, 3, 4, 56, 78, 99, 102]
data = 56

if bin_search(testList, data):
    print('yes')
else:
    print('no')


