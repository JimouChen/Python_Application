"""
# @Time    :  2020/8/26
# @Author  :  Jimou Chen
"""
import re

from bs4 import BeautifulSoup
import requests
import pandas as pd


class Crawler:
    def __init__(self):
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36'
        self.headers = {"User-Agent": self.user_agent}
        self.queue = set()  # 用集合做url队列可以自动去掉重复的url
        self.processed = set()
        self.save_list = []

    def crawl(self, url):

        if url not in self.processed:
            self.processed.add(url)  # 已经访问过的url

            resp = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(resp.content.decode('UTF-8'), 'html.parser')
            nodes = soup.find_all('div', {'class': 'hd'})

            # 当前页
            for i in nodes:
                name = i.find_all('span')[0].text
                link = i.find('a')['href']
                msg_dict = {'电影名': name, '链接': link}
                self.save_list.append(msg_dict)

                print('{}: {}'.format(i.find_all('span')[0].text, i.find('a')['href']))
            print('\n')
            # 其他页
            other_pages = soup.find_all('div', {'class': 'paginator'})
            # 正则表达式找到页码链接
            pattern = re.compile(r'<a href="(.+?)">\w')
            links = pattern.findall(str(other_pages))

            for link in links:
                url = 'https://movie.douban.com/top250' + link
                self.queue.add(url)

    def run(self, url):
        self.queue.add(url)
        # 保持获取网页，直到为队列为空
        while self.queue:
            self.crawl(self.queue.pop())

        # 这里可以保存到指定文件
        movie_msg = pd.DataFrame(self.save_list)
        movie_msg.to_excel('douban_movies.xls')
        print(self.save_list)


if __name__ == '__main__':
    Crawler().run('https://movie.douban.com/top250')
