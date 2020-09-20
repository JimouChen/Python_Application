"""
# @Time    :  2020/9/20
# @Author  :  Jimou Chen
"""
n = int(input())
two = bin(n)[2:]
rev = two[::-1]
print(int(rev, 2))