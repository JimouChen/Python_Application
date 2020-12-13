"""
# @Time    :  2020/12/13
# @Author  :  Jimou Chen
"""


# 得到三元组列表
def handle_file(file):
    words_list = []
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            init_step = line[:64]
            move_step = line[65:64 + 5]
            num = int(line[64 + 7:-1])
            word_tuple = (init_step, move_step, num)
            words_list.append(word_tuple)
            # print(word_tuple)
            line = f.readline()

    return words_list


# 返回下一步要下哪一步
def next_move(triad: list, init_input: str):
    for each in triad:
        if each[0] == init_input:
            return each[1]


if __name__ == '__main__':
    triad_list = handle_file('test.txt')
    print(triad_list)
    test = '0919293949596979891477866646260600102030405060708012720323436383'
    res = next_move(triad_list, test)
    print(res)
