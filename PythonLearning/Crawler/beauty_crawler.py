"""
# @Time    :  2020/7/23
# @Author  :  Jimou Chen
"""
import re
import requests

'''爬取多页图片的情况'''
x = 1


def get_html(url):
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
    headers = {'User-Agent': user_agent}
    respond = requests.get(url=url, headers=headers)
    return respond.text


# 得到图片的链接
def get_photo(html):
    pattern = re.compile(r'<img src="(.+?.jpeg)" title=')
    photo = pattern.findall(html)
    link_photo = []
    for each in photo:
        each = 'https:'+each
        link_photo.append(each)
    return link_photo


# 保存图片
def save_photo(photo_list):
    path = r'D:\图片\IDEA_PYCharm_CLion\crawler_test\pic'
    for photo in photo_list:
        resp = requests.get(photo)
        img = resp.content
        global x
        save_path = '{}{}.jpeg'.format(path, x)
        with open(save_path, 'wb') as f:
            f.write(img)
        x += 1


if __name__ == '__main__':

    # 第一页单独爬
    url_1 = 'https://jbh.17qq.com/article/qhwwhnpsy.html'
    html = get_html(url_1)
    photo_list = get_photo(html)
    print(photo_list)
    save_photo(photo_list)

    # 保存第二页到第6页的图片
    for i in range(2, 7):
        url = 'https://jbh.17qq.com/article/qhwwhnpsy_p' + str(i) + '.html'
        html = get_html(url)
        photo_list = get_photo(html)
        save_photo(photo_list)
        print(photo_list)

