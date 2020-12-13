"""
# @Time    :  2020/12/11
# @Author  :  Jimou Chen
"""
import json


def handle_json(json_file):
    dict_file = json.loads(json_file)
    init_list = []
    move_list = []
    for each_dict in dict_file:
        init_list.append(each_dict['init'])
        move_list.append(each_dict['move_list'])

    return init_list, move_list


# 得到下一个状态,返回(statues, move)格式
def handle_next(init_data: str, move_res: list):
    res = []
    for each in move_res:
        res.append((init_data, each))
        init_data = init_data.replace(each[0:2], each[2:])

    return res


# 切分每一步
def split_move(move: str):
    res = []
    for i in range(len(move)):
        if (i + 1) % 4 == 0:
            res.append(move[i - 3:i + 1])

    return res


if __name__ == '__main__':
    # test = '123456781012'
    # l = split_move(test)
    # res = handle_next('9999129999109956999912341234', l)
    # print(res)
    with open('chess.json') as f:
        json_file = f.read()
    init_list, move_list = handle_json(json_file)
    # print(init_list)
    # print(move_list)

    final_res = []
    # 切分步数
    for i in range(len(init_list)):
        move_res = split_move(move_list[i])
        map_key = handle_next(init_list[i], move_res)
        final_res.append(map_key)

    res_set = []
    for i in final_res:
        for j in i:
            res_set.append(j)

    print(res_set)
    # 保存成txt
    with open('res.txt', 'w+') as p:
        for i in res_set:
            p.write(i[0] + ',')
            p.write(i[1] + '\n')

'''
init:
09192939495969798917778666462606
00102030405060708072128363432303

0919293949596979891777866646260600102030405060708012720323436383
8979695949392919097717866646260600102030405060708012720323436383
8979695949392919097717866646260600102030405060708012720323436383
'''

'''

test

  {
    "init": "8699999949992999859987612070633158996499509942474099994638999999",
    "move_list": "61604260706050518581515281825251828151528682648281825251828151526362524287825852825247668183402083434232433332425259202733382747384847484948"
  },
  {
    "init": "8699999949992999859987612070633158996499509942474099994638999999",
    "move_list": "61604260706050518581515281825251828151528682648281825251828151526362524287825852825247668183402083434232433332425259202733382747384847484948"
  },
  {
    "init": "8699999949992999859987612070633158996499509942474099994638999999",
    "move_list": "61604260706050518581515281825251828151528682648281825251828151526362524287825852825247668183402083434232433332425259202733382747384847484948"
  },
'''