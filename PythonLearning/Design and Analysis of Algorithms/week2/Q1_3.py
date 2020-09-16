"""
# @Time    :  2020/9/17
# @Author  :  Jimou Chen
"""


def get_nearest_num(array: list):
    array.sort()
    temp_list = []
    for i in range(len(array) - 1):
        temp_list.append(array[i + 1] - array[i])

    min_index = temp_list.index(min(temp_list))
    index1 = min_index
    index2 = min_index + 1

    return array[index1], array[index2]


if __name__ == '__main__':
    l = [21, 223, 2, 45, 11, 44, 56]
    print(get_nearest_num(l))
