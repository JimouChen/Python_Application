"""
# @Time    :  2020/9/15
# @Author  :  Jimou Chen
"""
from itertools import combinations


def subsets(nums):
    all_subsets = []
    for i in range(len(nums) + 1):
        temp_list = combinations(nums, i)
        for each in temp_list:
            all_subsets.append(list(each))

    return all_subsets


my_list = input().split(',')
for i in range(len(my_list)):
    my_list[i] = int(my_list[i])

print(subsets(my_list))
