"""
# @Time    :  2020/8/17
# @Author  :  Jimou Chen
"""
import json


# 获取口座记号和口座番号
def get_mouth_mark_msg(delta_x, delta_y, prop_x, prop_y):
    mouth_mark = ''
    # 计算我们要的信息的坐标范围
    mouth_mark_x = [x_min + delta_x - 30, x_min + delta_x + prop_x * (x_max - x_min) + 20]
    mouth_mark_y = [y_min + delta_y - 5, y_min + delta_y + prop_y * (y_max - y_min) + 25]
    print(mouth_mark_x, mouth_mark_y)

    # 接下来根据坐标范围找出符合这个范围的数据
    for point_msg in json_res['textAnnotations']:
        # 每个矩形的四个顶点坐标列表
        point = point_msg['boundingPoly']['vertices']

        if 'x' not in point[0] and 'x' not in point[3]:
            continue

        # 找出符合坐标范围的坐标，得到对应的文字信息
        if point[0]['x'] >= mouth_mark_x[0] and point[1]['x'] <= mouth_mark_x[1] and \
                point[1]['y'] >= mouth_mark_y[0] and point[2]['y'] <= mouth_mark_y[1]:
            # 通过观察，发现口座番号至少5位，口座记号6位
            if len(point_msg['description']) >= 5:
                mouth_mark = point_msg['description']
                need_msg.append(mouth_mark)

    mouth_mark_msg.append(mouth_mark)
    print(need_msg)
    return mouth_mark


# 获取金额
def get_amount_msg(delta_x, delta_y, prop_x, prop_y):
    amount = ''
    # 计算金额的范围,通过右上角和右下角固定的坐标确定范围
    money_x = [x_max - delta_x - prop_x * (x_max - x_min) - 25, x_max - delta_x + 10]
    money_y = [y_max - delta_y - prop_y * (y_max - y_min) - 30, y_max - delta_y + 40]
    print(money_x, money_y)

    # 接下来根据坐标范围找出符合这个范围的数据
    for point_msg in json_res['textAnnotations']:
        # 每个矩形的四个顶点坐标列表
        point = point_msg['boundingPoly']['vertices']
        if 'x' not in point[0] and 'x' not in point[3]:
            continue
        # 找出符合坐标范围的坐标，得到对应的文字信息
        if point[0]['x'] >= money_x[0] and point[1]['x'] <= money_x[1] and \
                point[1]['y'] >= money_y[0] and point[2]['y'] <= money_y[1]:
            # 金额位数大于等于4位数，且最大位数字是不为0的数字,
            if 4 <= len(point_msg['description']) <= 8 and '1' <= point_msg['description'][0] <= '9':
                amount = point_msg['description']
                need_msg.append(amount)
                amount_msg.append(amount)

    print(need_msg)
    return amount


# 获取加入者名
def get_joiner_name(delta_x, delta_y, prop_x, prop_y):
    name = ''
    # 获取第一个字的左上下角和最后一个字的右下角坐标来算一整列文字信息的坐标范围
    name_x = [x_min + delta_x - 10, x_min + delta_x + prop_x * (x_max - x_min) + 10]
    name_y = [y_min + delta_y - 5, y_min + delta_y + prop_y * (y_max - y_min) + 15]
    print(name_x, name_y)
    # 接下来根据坐标范围找出符合这个范围的数据
    for point_msg in json_res['textAnnotations']:
        # 每个矩形的四个顶点坐标列表
        point = point_msg['boundingPoly']['vertices']
        if 'x' not in point[0] and 'x' not in point[3]:
            continue
        # 找出符合坐标范围的坐标，得到对应的文字信息
        if point[0]['x'] >= name_x[0] and point[1]['x'] <= name_x[1] and \
                point[1]['y'] >= name_y[0] and point[2]['y'] <= name_y[1]:
            # 把每个符合这个范围的字拼接起来
            name += point_msg['description']

    need_msg.append(name)
    joiner_name.append(name)
    print(need_msg)
    return name


# 去除字符串非数字的部分,保留数字部分
def save_digits(string):
    digits_str = ''
    for i in string:
        if '0' <= i <= '9':
            digits_str += i
    return digits_str


'''对得到的一式两份的值做对比，选出识别效果最佳的'''


# 处理口座记号,口座番号,确保识别得到的尽可能是纯数字
def handle_mark(res1, res2, n):
    # 先把里面的非数字部分去掉
    res1 = save_digits(res1)
    res2 = save_digits(res2)
    # 计算数字长度
    len1 = len(res1)
    len2 = len(res2)
    # 取数字长度最接近n,口座记号按照6位处理，口座番号按照
    if abs(len1 - n) > abs(len2 - n):
        return res2
    elif abs(len1 - n) < abs(len2 - n):
        return res1
    # 一样长度的情况
    elif n == 6:
        zero_num1 = res1.count('0')
        zero_num2 = res2.count('0')
        return res1 if zero_num1 >= zero_num2 else res2
    # 其他情况经过识别发现，左边的口座记号(res1)的识别效果更好
    else:
        return res1


# 处理金额的数据，
def handle_account(account_list):
    # 防止只写了一处金额或者没写金额的情况
    if len(account_list) == 1:
        return account_list[0]
    if len(account_list) == 0:
        return None

    # 先把里面的非数字部分去掉
    res1 = save_digits(account_list[0])
    res2 = save_digits(account_list[1])
    # 计算金额长度
    len1 = len(res1)
    len2 = len(res2)
    # 判断条件是金额至少是2位数，考虑到可能识别成字母，所以取位数较小者
    if (len2 <= len1 <= 8) and (len2 >= 2):
        return res2
    else:
        return res1


# 设置一个列表存放需要的所有信息
need_msg = []
mouth_mark_msg = []
amount_msg = []
joiner_name = []

with open("json_rsp/res6.json", 'r', encoding='UTF-8') as f:
    json_res = json.load(f)

# 照片四个顶点的坐标
points = json_res['textAnnotations'][0]['boundingPoly']['vertices']
print(points)

# 有些没有返回照片顶点x坐标的处理，取起点为0
if 'x' not in points[0]:
    points[0]['x'] = 0
if 'x' not in points[3]:
    points[3]['x'] = 0

# 先找出这张图片左上角和右下角的坐标
x_min = points[0]['x']
y_min = points[1]['y']
x_max = points[1]['x']
y_max = points[2]['y']

print('x的范围：', x_min, '--->', x_max)
print('y的范围：', y_min, '--->', y_max)

'''找左上角处的口座记号'''

# 设置两个变量分别表示左上角坐标到口座记号左上角坐标的距离
delta_x = 10
delta_y = 35
# 口座记号矩形长宽占这种图片长宽大小的比例
prop_x = 0.2  # 0.18
prop_y = 0.0627  # 0.0627

# 找左边的口座记号
key_msg = get_mouth_mark_msg(delta_x, delta_y, prop_x, prop_y)
print(key_msg)

'''找右边的口座记号'''
# 下面同理
delta_x = 475
delta_y = 35
prop_x = 0.2635
prop_y = 0.0627

key_msg = get_mouth_mark_msg(delta_x, delta_y, prop_x, prop_y)
print(key_msg)

'''左边的口座番号'''
delta_x = 157
delta_y = 39
prop_x = 0.185  # 0.1637
prop_y = 0.0627  # 0.05

key_msg = get_mouth_mark_msg(delta_x, delta_y, prop_x, prop_y)
print(key_msg)

'''右边的口座番号'''
delta_x = 513
delta_y = 69
prop_x = 0.1637
prop_y = 0.0627  # 0.05

key_msg = get_mouth_mark_msg(delta_x, delta_y, prop_x, prop_y)
print(key_msg)

'''左边金额'''
delta_x = 197
delta_y = 255
prop_x = 0.2062
prop_y = 0.0627  # 0.05

key_msg = get_amount_msg(delta_x, delta_y, prop_x, prop_y)
print(key_msg)

'''右边金额'''
delta_x = 5
delta_y = 166
prop_x = 0.2258
prop_y = 0.0627  # 0.05

key_msg = get_amount_msg(delta_x, delta_y, prop_x, prop_y)
print(key_msg)

'''加入者名'''
delta_x = 41
delta_y = 64
prop_x = 0.33879
prop_y = 0.0627  # 0.053

key_msg = get_joiner_name(delta_x, delta_y, prop_x, prop_y)
print(key_msg)

print('口座记号: ', mouth_mark_msg[:2])
print('口座番号: ', mouth_mark_msg[2:])
print('金额: ', amount_msg)
print('加入者名: ', joiner_name, '\n')

print('===========经过优化处理后的数据============')
print('口座记号: ', handle_mark(mouth_mark_msg[0], mouth_mark_msg[1], 6))
print('口座番号: ', handle_mark(mouth_mark_msg[2], mouth_mark_msg[3], 5))
print('金额: ', handle_account(amount_msg))
print('加入者名: ', joiner_name[0])
