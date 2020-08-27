"""
# @Time    :  2020/8/27
# @Author  :  Jimou Chen
"""
import scrapy
from bs4 import BeautifulSoup


class MassageSpider(scrapy.Spider):
    name = 'msg_crawl'  # 爬虫的名字，一定要给
    start_urls = ['http://www.cae.cn/cae/html/main/col48/column_48_1.html']  # 起始的url

    # 对爬到的网页进行解析
    def parse(self, response, **kwargs):
        soup = BeautifulSoup(response.body, 'html.parser')
        nodes = soup.find_all('li', {'class': 'name_list'})
        i = 0
        for node in nodes:
            i += 1
            people_name = node.find('a').text
            link = 'http://www.cae.cn/' + node.find('a')['href']
            print('{}. {}: {}'.format(i, people_name, link))
