"""
# @Time    :  2020/7/22
# @Author  :  Jimou Chen
"""
import re
import requests


def get_html_text(url):
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
    headers = {'User-Agent': user_agent}
    # 发送请求
    respond = requests.get(url=url, headers=headers)
    html_text = respond.text

    return html_text


def match_text(html_text):
    # 需要得到的内容，在  .+?  两边一定要加括号
    pattern = re.compile(r'')
    msg = pattern.findall(html_text)

    return msg


url = ''
html_text = get_html_text(url)
massage = match_text(html_text)

print(massage)