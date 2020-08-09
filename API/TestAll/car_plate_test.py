"""
# @Time    :  2020/8/9
# @Author  :  Jimou Chen
"""
from aip import AipImageClassify

""" 你的 APPID AK SK """
APP_ID = '21889084'
API_KEY = 'bH5eTQlo01yNGD6RXoXFMQba'
SECRET_KEY = 'woFkuNWEwWr2PyG6ToQUSzffAjswSW8b'

client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)

""" 读取图片 """


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


image = get_file_content('car.jpg')

""" 调用车辆识别 """
client.carDetect(image);

""" 如果有可选参数 """
options = {}
options["top_num"] = 3
options["baike_num"] = 5

""" 带参数调用车辆识别 """
image = client.carDetect(image, options)
print(image)


# msg = {'log_id': 4152560610687609961,
#        'location_result': {'width': 1194.911865234375, 'top': 471.3589477539062, 'height': 877.8688354492188,
#                            'left': 470.4656677246094},
#        'result': [{'score': 0.9967948198318481, 'year': '2018', 'baike_info': {}, 'name': '丰田荣放(RAV4)'},
#                   {'score': 0.0002847549039870501, 'year': '2013-2017', 'baike_info': {
#                       'baike_url': 'http://baike.baidu.com/item/%E4%B8%B0%E7%94%B0%E8%8A%B1%E5%86%A0/10372834',
#                       'image_url': 'https://bkimg.cdn.bcebos.com/pic/c2fdfc039245d688fb491670aac27d1ed31b249e',
#                       'description': '花冠(COROLLA)是丰田汽车旗下的老牌产品，于1966年在日本下线，寓意“花中之冠”。1970年、1974年、1979年、1983年、1987年、1991年、1995年和2000年分别推出新一代花冠，已经是第9代车型。丰田花冠全球累计销量超过3600万辆，达成单一品牌累计销量总冠军。第九代花冠(Corolla EX)由一汽丰田生产，是丰田国产车型中首款采用VVT-i发动机的车型。丰田第10代COROLLA轿车将中文名从“花冠”改为“卡罗拉”。'},
#                    'name': '丰田花冠'},
#                   {'score': 0.0001011483400361612, 'year': '2012-2017', 'baike_info': {
#                'baike_url': 'http://baike.baidu.com/item/%E9%9B%AA%E4%BD%9B%E5%85%B0%E7%A7%91%E5%B8%95%E5%A5%87/3959055',
#                'image_url': 'https://bkimg.cdn.bcebos.com/pic/42a98226cffc1e17622bb3664590f603728de9d5',
#                'description': '雪佛兰科帕奇是雪佛兰推出的一款车。雪佛兰正式宣布2015款科帕奇的上市价格，新车共推出4款车型，售价为17.99万~20.99万元，目前全国雪佛兰经销商店已全面接受预订。'},
#                                      'name': '雪佛兰科帕奇'}], 'color_result': '棕色'}
#

for i in range(len(image['result'])):
    product_year = image['result'][i]['year']
    name = image['result'][i]['name']
    print(product_year)
    print(name)
color = image['color_result']

print(color)
