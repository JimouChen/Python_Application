"""
# @Time    :  2020/12/13
# @Author  :  Jimou Chen
"""
import pandas as pd
import numpy as np


# 得到三元组列表
def handle_file(file):
    words_list = []
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            words_dict = {}
            init_step = line[:64]
            move_step = str(line[65:64 + 5])
            num = int(line[64 + 7:-1])
            word_tuple = (init_step, move_step, num)
            words_dict['init'], words_dict['move'], words_dict['num'] = init_step, move_step, num
            # words_list.append(word_tuple)
            words_list.append(words_dict)
            # print(word_tuple)
            line = f.readline()

    return words_list


# 返回下一步要下哪一步
def next_move(triad: list, init_input: str):
    for each in triad:
        if each[0] == init_input:
            return each[1]


# 存到csv文件
# def save_to_csv(txt_file):
#     dataset = pd.read_table(txt_file, sep=' ', header=0)
#     dataset.to_excel("w.xls", index=False)  # 保存为csv格式
    # data_txt = np.loadtxt(txt_file)
    # data_txtDF = pd.DataFrame(data_txt)
    # data_txtDF.to_csv('w.csv', index=False)


if __name__ == '__main__':
    triad_list = handle_file('word_count.txt')
    # print(triad_list)
    df = pd.DataFrame(triad_list)
    df.to_excel('word_count.xlsx', index=True)

    # # print(len(triad_list))
    #
    # input_init = '5772472939576565654158716657158714144646804158728571515714685858'
    # res = next_move(triad_list, input_init)
    # print(res)
    # save_to_csv('test.txt')
