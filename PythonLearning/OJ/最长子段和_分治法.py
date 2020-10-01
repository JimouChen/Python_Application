"""
# @Time    :  2020/9/30
# @Author  :  Jimou Chen
"""


def max_sum(l: list, left, right):
    if left == right:
        return l[left]
    else:
        mid = (left + right) // 2
        left_sum = max_sum(l, left, mid)
        right_sum = max_sum(l, mid + 1, right)

        # 处理中间的
        s1 = 0
        s1_max = 0
        # 从中间到左边
        for i in range(mid, left - 1, -1):
            s1 += l[i]
            if s1 > s1_max:
                s1_max = s1

        s2 = 0
        s2_max = 0
        # 从中间到右边
        for i in range(mid + 1, right):
            s2 += l[i]
            if s2_max < s2:
                s2_max = s2

        s_max = s2_max + s1_max
        return max(left_sum, right_sum, s_max)


# ll = [1, -2, 3, -4, 5, 6, -7, 4, 3, -3, 1]
ll = [-2, 11, -4, 13, -5, -2]
a = max_sum(ll, 0, len(ll) - 1)
print(a)
