"""
# @Time    :  2020/8/26
# @Author  :  Jimou Chen
"""
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin


class Crawler:
    def __init__(self):
        self.queue = set()  # 存放抓取的url
        self.processed = set()  # 存放已经处理过的url

    def crawl(self, url):
        if url not in self.processed:
            resp = requests.get(url)
            soup = BeautifulSoup(resp.content.decode('UTF-8'), 'html.parser')
            # 第一页的
            for i in soup.find('div', {'class': 'listList'}).find_all('li'):
                print('{}: {}'.format(i.find_all('span')[1].text, i.a.text))
            # 其他页的
            for i in soup.find_all('span', {'class': 'p_no'}):
                self.queue.add(urljoin(url, i.a.get('href')))
            self.processed.add(url)

    def run(self, url):
        self.queue.add(url)
        while self.queue:
            self.crawl(self.queue.pop())


if __name__ == '__main__':
    Crawler().run('http://www.dgut.edu.cn/xwzx/ggyw.htm')