"""
# @Time    :  2020/7/26
# @Author  :  Jimou Chen
"""


def input_n_array(n):
    x_str_list = input().split(' ')
    for i in range(n):
        x_str_list[i] = int(x_str_list[i])

    return x_str_list


while True:
    try:
        n, m = map(int, input().split())
        l = input_n_array(n)
        l.sort(reverse=True)
        for i in range(m):
            if i < m-1:
                print(l[i], end=' ')
            else:
                print(l[i])


    except:
        break
