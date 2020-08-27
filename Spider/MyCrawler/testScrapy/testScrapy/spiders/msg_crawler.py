"""
# @Time    :  2020/8/27
# @Author  :  Jimou Chen
"""
import scrapy
from bs4 import BeautifulSoup


class MassageSpider(scrapy.Spider):
    name = 'msg_crawl'  # 爬虫的名字，一定要给
    # start_urls = ['http://www.cae.cn/cae/html/main/col48/column_48_1.html']  # 起始的url
    start_urls = ['http://quotes.toscrape.com/page/1/']  # 起始的url

    page_num = 1

    # 对爬到的网页进行解析
    def parse(self, response, **kwargs):
        soup = BeautifulSoup(response.body, 'html.parser')
        # nodes = soup.find_all('li', {'class': 'name_list'})
        # i = 0
        # for node in nodes:
        #     i += 1
        #     people_name = node.find('a').text
        #     link = 'http://www.cae.cn/' + node.find('a')['href']
        #     print('{}. {}: {}'.format(i, people_name, link))

        nodes = soup.find_all('div', {'class': 'quote'})
        for node in nodes:
            word = node.find('span', {'class': 'text'}).text
            people = node.find('small', {'class': 'author'}).text
            print('{0:<4}: {1:<20} said: {2:<20}'.format(self.page_num, people, word))

        self.page_num += 1
        try:
            url = soup.find('li', {'class': 'next'}).a['href']
            if url is not None:
                next_link = 'http://quotes.toscrape.com' + url
                yield scrapy.Request(next_link, callback=self.parse)
        except Exception:
            print('所有页面爬取结束！')

    # def start_requests(self):
    #     url = 'http://quotes.toscrape.com/'
    #
    #     tag = getattr(self, 'tag', None)
    #     if tag is not None:
    #         url = url + 'tag/' + tag
    #     yield scrapy.Request(url, self.parse)
    #
    # def parse(self, response):
    #     for quote in response.css('div.quote'):
    #         yield {
    #             'text': quote.css('span.text::text').get(),
    #             'author': quote.css('small.author::text').get(),
    #         }
    #
    #     next_page = response.css('li.next a::attr(href)').get()
    #     if next_page is not None:
    #         yield response.follow(next_page, self.parse)
