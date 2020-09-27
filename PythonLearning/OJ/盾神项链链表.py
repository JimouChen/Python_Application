"""
# @Time    :  2020/9/24
# @Author  :  Jimou Chen
"""
n, m = map(int, input().split())
link = list(map(int, input().split()))
while m:
    m -= 1
    zl = list(input().split(' '))
    if 'DEL' in zl:
        link.remove(int(zl[1]))
    if 'ADD' in zl:
        p_index = link.index(int(zl[1]))
        link.insert(p_index, int(zl[2]))

l = len(link)
print(l)
for i in range(l):
    if i != l - 1:
        print(link[i], end=' ')
    else:
        print(link[i])
