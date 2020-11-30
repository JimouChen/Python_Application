"""
# @Time    :  2020/11/30
# @Author  :  Jimou Chen
"""
# s = input()
# l = 0
# r = 0
# max_ = 0
# for i in range(len(s)):
#     if s[i] == '(':
#         l += 1
#     else:
#         r += 1
#     if l == r:
#         max_ = max(max_, 2 * r)
#     elif r >= l:
#         l = r = 0
#
# l = r = 0
# i = len(s) - 1
# while i >= 0:
#     if s[i] == '(':
#         l += 1
#     else:
#         r += 1
#
#     if l == r:
#         max_ = max(max_, 2 * l)
#     elif l >= r:
#         l = r = 0
#     i -= 1
#
# print(max_, end='')

def pd(s):
    if not s:
        return 0

    stack = []
    ans = 0
    for i in range(len(s)):
        # 入栈条件
        if not stack or s[i] == '(' or s[stack[-1]] == ')':
            stack.append(i)  # 下标入栈
        else:
            # 计算结果
            stack.pop()
            ans = max(ans, i - (stack[-1] if stack else -1))
    return ans

ss = input()
print(pd(ss))