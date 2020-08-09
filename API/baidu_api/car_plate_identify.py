"""
# @Time    :  2020/8/9
# @Author  :  Jimou Chen
"""
from aip import AipOcr
import time

time_start = time.time()
""" 你的 APPID AK SK """
APP_ID = '21893106'
API_KEY = 'whTDUUyAzgIP6pqkmRzBFy1c'
SECRET_KEY = '3LtSVptmrrTHtZffk9uGKagsMSoxvzkl'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

""" 读取图片 """


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


image = get_file_content('car.jpg')

""" 调用车牌识别 """
client.licensePlate(image)

""" 如果有可选参数 """
options = {"multi_detect": "true"}

""" 带参数调用车牌识别 """
res = client.licensePlate(image, options)
print(res)


print('车牌号：', res['words_result'][0]['number'])
print('颜色：', res['words_result'][0]['color'])

time_end = time.time()
print('执行时间：', time_end - time_start, 's')