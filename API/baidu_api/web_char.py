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


image = get_file_content('image/car6.jpg')

""" 调用网络图片文字识别, 图片参数为本地图片 """
# client.webImage(image)

""" 如果有可选参数 """
options = {}
options["detect_direction"] = "true"
options["detect_language"] = "true"

""" 带参数调用网络图片文字识别, 图片参数为本地图片 """
res = client.webImage(image, options)
print(res['words_result'])
# url = "https//www.x.com/sample.jpg"
#
# """ 调用网络图片文字识别, 图片参数为远程url图片 """
# client.webImageUrl(url);
#
# """ 如果有可选参数 """
# options = {}
# options["detect_direction"] = "true"
# options["detect_language"] = "true"
#
# """ 带参数调用网络图片文字识别, 图片参数为远程url图片 """
# client.webImageUrl(url, options)

time_end = time.time()
print('执行时间：', time_end - time_start, 's')