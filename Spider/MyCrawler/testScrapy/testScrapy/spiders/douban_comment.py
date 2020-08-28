"""
# @Time    :  2020/8/28
# @Author  :  Jimou Chen
"""
import scrapy
from bs4 import BeautifulSoup


class CommentSpider(scrapy.Spider):
    name = 'comment_spider'
    start_urls = ['https://book.douban.com/subject/35092383/annotation']
    custom_settings = {
        "USER_AGENT": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36',
    }
    page_num = 1

    def parse(self, response, **kwargs):
        soup = BeautifulSoup(response.body, 'html.parser')
        nodes = soup.find_all('div', {'class': 'short'})

        print('======================{}======================'.format(self.page_num))
        for node in nodes:
            comment = node.find('span').text
            print(comment, end='\n\n')
        self.page_num += 1

        # 其他页链接
        num = 10 * self.page_num
        if self.page_num <= 28:
            url = 'https://book.douban.com/subject/35092383/annotation?sort=rank&start=' + str(num)
            yield scrapy.Request(url, callback=self.parse)
