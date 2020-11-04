"""
# @Time    :  2020/10/14
# @Author  :  Jimou Chen
"""

num = [0 for i in range(100)]
temp = [0 for i in range(100)]
count = 0


def merge(low, high, mid):
    left = low  # 左边数组指针
    right = mid + 1  # 右边数组指针
    k = low  # temp数组指针
    global count

    while left <= mid and right <= high:
        if num[left] > num[right]:
            temp[k] = num[right]
            k += 1
            right += 1
            # 求逆序对
            count += mid - left + 1
        else:
            temp[k] = num[left]
            k += 1
            left += 1

    # 检测左边
    while left <= mid:
        temp[k] = num[left]
        k += 1
        left += 1

    # 检查右边
    while right <= high:
        temp[k] = num[right]
        k += 1
        right += 1

    # 拷贝
    for i in range(low, high + 1):
        num[i] = temp[i]


def merge_sort(low, high):
    if low >= high:
        return

    # 分
    mid = (high + low) // 2
    # mid = low + (high - low) // 2
    merge_sort(low, mid)
    merge_sort(mid + 1, high)

    # 治
    merge(low, high, mid)


if __name__ == '__main__':
    # array = [3, 5, 2, 4, 6]
    # 输入
    num = list(map(int, input().split()))
    merge_sort(0, len(num) - 1)
    print(count)

'''
3 5 2 4 6

3
'''