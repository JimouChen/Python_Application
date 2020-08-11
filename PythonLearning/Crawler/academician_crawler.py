"""
# @Time    :  2020/7/23
# @Author  :  Jimou Chen
"""
import re
import requests

'''爬取中科院院士名单'''


def get_html(url):
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
    headers = {'User-Agent': user_agent}
    respond = requests.get(url=url, headers=headers)
    # respond.encoding = 'gb2312'
    return respond.text


def get_msg(html):
    pattern = re.compile(r'target="_blank">(.+?)</a></li>')
    name_list = pattern.findall(html)

    return name_list


if __name__ == '__main__':
    url = 'http://www.cae.cn/cae/html/main/col48/column_48_1.html'
    html_txt = get_html(url)
    name_list = get_msg(html_txt)
    print(name_list)
    print(len(name_list))
