"""
# @Time    :  2020/9/17
# @Author  :  Jimou Chen
"""


def fbn(n):
    if n == 1 or n == 2:
        return 1
    else:
        return fbn(n - 2) + fbn(n - 1)


input_list = input().split(' ')
for i in range(len(input_list)):
    input_list[i] = int(input_list[i])

temp = []
sum1 = 0
for i in range(1, input_list[0] + 1):
    sum1+=fbn(i)
# temp_all = sum(temp)
res = sum1 % fbn(input_list[1]) % input_list[2]
print(res)
