"""
# @Time    :  2020/9/23
# @Author  :  Jimou Chen
"""
from itertools import permutations

a = [0, 1, 2]
per = permutations(a, 3)
all_case = []
for i in per:
    all_case.append(list(i))

task1 = [9, 2, 7]
task2 = [6, 4, 3]
task3 = [5, 8, 1]

all_value = []
for each in all_case:
    value = task1[each[0]] + task2[each[1]] + task3[each[2]]
    all_value.append(value)

print(all_value)
print(min(all_value))