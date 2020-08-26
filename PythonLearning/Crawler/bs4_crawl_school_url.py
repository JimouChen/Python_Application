"""
# @Time    :  2020/8/26
# @Author  :  Jimou Chen
"""
from bs4 import BeautifulSoup
import requests


class Crawler:
    def __init__(self, url):
        self.url = url

    def crawl(self):
        print('Crawling...:{}'.format(self.url))
        resp = requests.get(self.url)
        soup = BeautifulSoup(resp.content.decode('UTF-8'), 'html.parser')
        nodes = soup.find('div', {'class': 'listList'}).find_all('li')
        for node in nodes:
            msg = node.a.text
            time = node.find_all('span')[1].text
            print('{}:{}'.format(time, msg))


if __name__ == '__main__':
    c = Crawler('https://www.dgut.edu.cn/xwzx/ggyw.htm')
    c.crawl()
