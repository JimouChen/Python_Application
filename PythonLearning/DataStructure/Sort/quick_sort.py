"""
# @Time    :  2020/11/2
# @Author  :  Jimou Chen
"""
# 全局变量
array = []


# 传入最左边和最右边的索引
def quickSort(left, right):
    if left > right:
        return
    # 设置一个基准数,还有两个哨兵
    temp = array[left]
    i = left
    j = right
    while i != j:
        # 先j从右边到左边，再i从左到右走,里面的i一定是<j
        while array[j] >= temp and i < j:
            j -= 1
        while array[i] <= temp and i < j:
            i += 1

        # 当i，j没有相遇
        if i < j:
            array[i], array[j] = array[j], array[i]

    # 此时退出循环时i已经=j,即i的位置就是基准数正确的位置
    array[i], array[left] = array[left], array[i]
    # 对基准数正确的位置左右两边递归
    quickSort(left, i - 1)
    quickSort(i + 1, right)
    return


if __name__ == '__main__':
    n = int(input())
    array = list(map(int, input().split()))
    quickSort(0, n-1)
    print(array)

'''
6
6 2 7 3 9 8
'''