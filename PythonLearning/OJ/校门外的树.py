"""
# @Time    :  2020/10/28
# @Author  :  Jimou Chen
"""
num, n = map(int, input().split())
tree = [1 for _ in range(num + 1)]

for i in range(n):
    start, end = map(int, input().split())
    for j in range(start, end+1):
        if tree[j] == 1:
            tree[j] = 0

res = sum(tree)
print(res)