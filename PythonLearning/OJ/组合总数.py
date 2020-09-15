"""
# @Time    :  2020/9/15
# @Author  :  Jimou Chen
"""
'''
找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，
并且每种组合中不存在重复的数字。
说明：
所有数字都是正整数。
解集不能包含重复的组合。 
示例 1:
输入: k = 3, n = 7
输出: [[1,2,4]]
示例 2:
输入: k = 3, n = 9
输出: [[1,2,6], [1,3,5], [2,3,4]]
'''

from itertools import combinations

while True:

    try:
        out_list = []
        L = list(range(1, 10))
        k, n = map(int, input().split())
        all_cb = combinations(L, k)
        for each_cb in all_cb:
            each_cb = list(each_cb)
            if n == sum(each_cb):
                out_list.append(each_cb)

        print(out_list)
    except:
        break
