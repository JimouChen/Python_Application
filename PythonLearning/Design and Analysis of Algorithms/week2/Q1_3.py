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


def get_min_diff(array: list):
    array.sort()
    temp_list = []
    for i in range(len(array) - 1):
        diff = array[i + 1] - array[i]
        temp_list.append(diff)

    return min(temp_list)


if __name__ == '__main__':
    test = [21, 223, 2, 45, 11, 44, 56]
    print('最接近的两个数是:', get_nearest_num(test))
    print('最接近的两个数的差是:', get_min_diff(test))
