"""
# @Time    :  2020/10/1
# @Author  :  Jimou Chen
"""


def get_gray_code(n):
    # 初始化一个长度为2^n的列表
    code_list = ['' for i in range(0, 2 ** n)]
    if n == 1:
        code_list[0] = '0'
        code_list[1] = '1'
        return code_list

    last_list = get_gray_code(n - 1)

    for i in range(0, len(last_list)):
        code_list[i] = '0' + last_list[i]
        code_list[len(code_list) - i - 1] = '1' + last_list[i]

    return code_list


print(get_gray_code(3))
print(get_gray_code(4))
