"""
# @Time    :  2020/9/18
# @Author  :  Jimou Chen
"""
from itertools import permutations

n = int(input())
for c in range(0, n):

    nums = input().split(' ')
    for i in range(len(nums)):
        nums[i] = int(nums[i])

    string = ''
    cb = permutations(nums, 4)
    str_list = []
    f = 0
    for each in cb:
        each = list(each)

        for i in each:
            print(i, end='')
        f += 1
        if f % 6 == 0:
            print(end='\n')
        else:
            print(end=' ')
    if c != n-1:
        print()