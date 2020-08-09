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


image = get_file_content('example.jpg')

""" 调用通用物体识别 """
client.advancedGeneral(image)

""" 如果有可选参数 """
options = {}
options["baike_num"] = 5

""" 带参数调用通用物体识别 """
image = client.advancedGeneral(image, options)
print(image)

msg = {'log_id': 8430217777541641993, 'result_num': 5,
       'result': [{'score': 0.796593, 'root': '非自然图像-其他',
                   'baike_info': {
                       'baike_url': 'http://baike.baidu.com/item/%E8%A3%85%E4%BF%AE%E6%95%88%E6%9E%9C%E5%9B%BE/4676077',
                       'image_url': 'https://bkimg.cdn.bcebos.com/pic/730e0cf3d7ca7bcb157bebb4b4096b63f624a82c',
                       'description': '装修效果图是对设计师或装修客户的设计意图和构思进行形象化再现的形式设计师通过手绘或电脑软件在装修施工前就设计出房子装修后的风格效果的图。可以提前让客户知道以后装修是什么样子。 装修效果图分为室内装修效果图,室外装修效果图。一般装修层面来讲，室内装修效果图更多见。'},
                   'keyword': '装修效果图'},
                  {'score': 0.618507, 'root': '建筑-居家室内',
                   'baike_info': {}, 'keyword': '房间内景'},
                  {'score': 0.450976, 'root': '建筑-居家室内', 'baike_info': {
                      'baike_url': 'http://baike.baidu.com/item/%E5%AE%A4%E5%86%85/4022282',
                      'image_url': 'https://bkimg.cdn.bcebos.com/pic/342ac65c103853436d3f7a839713b07ecb8088a4'},
                   'keyword': '室内'},
                  {'score': 0.283788, 'root': '建筑-居家室内', 'baike_info': {
                      'baike_url': 'http://baike.baidu.com/item/%E7%94%B5%E8%A7%86%E8%83%8C%E6%99%AF%E5%A2%99/1806129',
                      'image_url': 'https://bkimg.cdn.bcebos.com/pic/a044ad345982b2b7efcf2a183badcbef77099b9d',
                      'description': '电视背景墙，是从公共建筑装修中引入的一个概念。它主要是指在客厅、办公室、卧室主要的有一面墙能反映自己的形象和风格。电视背景墙是居室背景墙装饰的重点之一，在背景墙设计中占据相当重要的地位，电视背景墙通常是为了弥补家居空间电视区背景墙的空旷，同时起到修饰电视区背景墙的作用。'},
                   'keyword': '电视背景墙'},
                  {'score': 0.116954, 'root': '商品-家具', 'baike_info': {
                      'baike_url': 'http://baike.baidu.com/item/%E5%BA%8A/13034502',
                      'image_url': 'https://bkimg.cdn.bcebos.com/pic/e850352ac65c10386752dd9cb5119313b07e89bb',
                      'description': '床一般在卧室、宿舍、病房、旅馆等场所使用，通常用以满足人类日常睡觉，记录监测体重，设计趋向智能。旧时通常以木材为材料，也可以不锈钢、金属为主要材料。一般有几十个零件组成。标配的以床头、床架、床尾、床腿、床板床垫、电动推杆、左右安全护挡、绝缘静音脚轮、一体餐桌、设备置物托盘为组件，采用冲孔、装配、焊接、压铆、除锈、喷塑等工艺。具有阻燃、防鼠、防蛀、耐用、实用简约、美观、易清洁、移动方便、使用安全等特点。人的三分之一的时间都是在床上度过的。 经过千百年的不断演化和改进设计，除了满足常人睡觉，还可以装上配套床垫、体重监测仪、气垫、医疗康复设备、尿袋隐形挂钩，也可增加就餐、电动起背、定时翻身、移动行走、站立、负压接尿、坐躺、取暖降温等其它功能，即可满足常人睡觉，也可满足儿童、重症病人、残障人士、老年人、医院病人、在家医疗康复、学习生活、办公应急等各种需要。床的种类有平板床、单人床、上下铺双层床、动力床、双人床、关节康复护理床、电动起背床、智能翻身护理床、残障床、四柱床、儿童床、病号床、治疗床、折叠床、体重监测护理床、按摩床、电动站床、气垫床、日床等。'},
                   'keyword': '床'}]}
