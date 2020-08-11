"""
# @Time    :  2020/8/11
# @Author  :  Jimou Chen
"""


def select_sort(arr):
    for i in range(len(arr)):
        max_index = 0
        for j in range(len(arr) - i):
            if arr[j] > arr[max_index]:
                max_index = j

        arr[len(arr) - i - 1], arr[max_index] = arr[max_index], arr[len(arr) - i - 1]


test = [8, 7, 6, 5, 43, 234, 57, 78, 2, 2, 54, 90, 1]
select_sort(test)
print(test)
