"""
# @Time    :  2020/7/22
# @Author  :  Jimou Chen
"""
import requests
import re

url = 'https://www.kugou.com/yy/html/rank.html'
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
headers = {'User-Agent': user_agent}

# 发送请求
respond = requests.get(url=url, headers=headers)
html = respond.text
pat = re.compile(r'class=" " title="(.+?)"')  # 需要得到的内容.+?一定要加括号
song_msg = pat.findall(html)

print(song_msg)
