"""
# @Time    :  2020/7/16
# @Author  :  Jimou Chen
"""
'''
自定义一个函数：传入n个整数，得到对应的列表，用于oj输入数组
'''


def input_n_array(n):
    x_str_list = input().split(' ')
    for i in range(n):
        x_str_list[i] = int(x_str_list[i])

    return x_str_list


'''
输入多组的情况
'''
while True:
    try:
        n = int(input())
        a = input_n_array(n)
        print(a)

    except:
        break


# 将输入的多个数放在列表里，模拟数组
def to_list(*args):
    my_list = [x for x in args]
    return my_list


l = to_list(2, 3, 4, 5, 9)
print(l)