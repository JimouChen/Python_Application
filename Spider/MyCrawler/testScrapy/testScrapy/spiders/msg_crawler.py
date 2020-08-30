"""
# @Time    :  2020/8/27
# @Author  :  Jimou Chen
"""
import scrapy
from bs4 import BeautifulSoup
import requests
from testScrapy.items import TestscrapyItem


class MassageSpider(scrapy.Spider):
    name = 'msg_crawl'  # 爬虫的名字，一定要给
    start_urls = ['http://quotes.toscrape.com/page/1/']  # 起始的url

    page_num = 1
    author_url = []

    # 对爬到的网页进行解析
    def parse(self, response, **kwargs):
        soup = BeautifulSoup(response.body, 'html.parser')
        nodes = soup.find_all('div', {'class': 'quote'})
        for node in nodes:
            word = node.find('span', {'class': 'text'}).text
            people = node.find('small', {'class': 'author'}).text
            tags = node.find_all('a', {'class': 'tag'})
            tags_list = []
            for i in range(len(tags)):
                tags_list.append(tags[i].text)
            # print('{} : '.format(self.page_num), tags_list)

            # print('{} : '.format(self.page_num), author_link)
            # 现在找到作者链接后，进去爬里面的数据信息
            author_link = 'http://quotes.toscrape.com/' + node.find_all('span')[1].a['href']
            yield response.follow(author_link, self.author_parse)
            item = TestscrapyItem(page=self.page_num, name=people, word=word, tags=tags_list)
            yield item
            # print('{0:<4}: {1:<20} said: {2:<20}\n{3}'.format(self.page_num, people, word, tags_list))

        print('=================================='*2 + 'ok' + '=================================='*2)
        self.page_num += 1
        try:
            url = soup.find('li', {'class': 'next'}).a['href']
            if url is not None:
                next_link = 'http://quotes.toscrape.com' + url
                yield scrapy.Request(next_link, callback=self.parse)
        except Exception:
            print('所有页面爬取结束！')

    def author_parse(self, response, **kwargs):
        soup = BeautifulSoup(response.body, 'html.parser')
        author = soup.find_all('div', {'class': 'author-details'})[0].find('h3').text
        birthday = soup.find('span').text
        bio = soup.find('div', {'class': 'author-description'}).text
        item = TestscrapyItem(author=author, birthday=birthday, bio=bio)
        yield item
        # print('{}: {}\n{}\n{}\n'.format(self.page_num, author, birthday, bio))
