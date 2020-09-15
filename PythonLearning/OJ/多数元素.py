"""
# @Time    :  2020/9/15
# @Author  :  Jimou Chen
"""


def get_input_list():
    array = input().split(',')
    for i in range(len(array)):
        array[i] = int(array[i])
    return array


a_list = get_input_list()
out = []
more_num = len(a_list) // 2
for each in a_list:
    if a_list.count(each) > more_num:
        out.append(each)

out = tuple(set(out))
for each in out:
    print(each, end=' ')


# 更快的解法
def num(nums: list[int]):
    nums.sort()
    return nums[len(nums) // 2]
