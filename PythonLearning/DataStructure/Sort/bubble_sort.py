"""
# @Time    :  2020/8/11
# @Author  :  Jimou Chen
"""


def bubble_sort(arr):
    for i in range(len(arr)):
        exchanged = False
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                exchanged = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
        # 如果已经排好，可以提前退出
        if not exchanged:
            return


test = [33, 0, 5, 2, 34, 5, 7, 66, 3, 12]
test1 = [1, 2, 3, 4, 12]
bubble_sort(test)
print(test)
