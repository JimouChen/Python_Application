"""
# @Time    :  2020/10/21
# @Author  :  Jimou Chen
"""
'''在指定范围a到b内折半查找，减治法'''
temp = []


def find_range(array: list, a, b, low, high):
    mid = (low + high) // 2
    # 左边找
    if array[mid] > b:
        find_range(array, a, b, low, mid)
    # 右边找
    if array[mid] < a:
        find_range(array, a, b, mid, high)
    # 夹在中间的情况
    else:
        i = mid
        while i >= low and array[i] >= a:
            # print(array[i], end=' ')
            temp.append(array[i])
            i -= 1

        j = mid + 1
        while j <= high and array[j] <= b:
            # print(array[j], end=' ')
            temp.append(array[j])
            j += 1


if __name__ == '__main__':
    test = [2, 5, 8, 13, 17, 25, 35, 36, 47, 50]
    find_range(test, 16, 36, 0, len(test))
    temp.sort()
    print(temp)
