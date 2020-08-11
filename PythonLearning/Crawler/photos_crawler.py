"""
# @Time    :  2020/7/23
# @Author  :  Jimou Chen
"""
import re
import requests
import urllib.request


def get_html(url):
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
    headers = {'User-Agent': user_agent}
    respond = requests.get(url=url, headers=headers)
    return respond.text


def get_photo(html):
    pattern = re.compile(r'<img src="(.+?.jpg)" alt=')
    photo = pattern.findall(html)
    return photo


# 保存图片
def save_photo(photo_list):
    path = r'D:\图片\IDEA_PYCharm_CLion\crawler_test\pic'
    x = 1
    for photo in photo_list:
        urllib.request.urlretrieve(photo, '{}{}.jpg'.format(path, x))
        x += 1


if __name__ == '__main__':
    url = 'https://www.tt98.com/tag/qingxinmeinv/'
    html_text = get_html(url)
    photo_list = get_photo(html_text)
    save_photo(photo_list)
    print(photo_list)
