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

    cb = permutations(nums, 4)
    for each in cb:
        each = list(each)
        j = 0
        # if j
        for j in range(6):
            if j < 5:
                print(each[j], end=' ')
            elif j == 5:
                print(each[j], end='\n')


