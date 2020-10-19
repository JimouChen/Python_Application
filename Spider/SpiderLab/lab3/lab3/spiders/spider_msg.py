import scrapy
from bs4 import BeautifulSoup
from lab3.items import Lab3Item


class QuoteSpider(scrapy.Spider):
    name = 'quotes'
    start_urls = ['http://quotes.toscrape.com/page/1/']
    page_num = 1

    # 对爬取到的信息进行解析
    def parse(self, response, **kwargs):
        soup = BeautifulSoup(response.body, 'html.parser')
        nodes = soup.find_all('div', {'class': 'quote'})
        for node in nodes:
            text = node.find('span', {'class': 'text'}).text
            author = node.find('small', {'class': 'author'}).text
            tags = node.find_all('a', {'class': 'tag'})
            tags_list = []
            for tag in tags:
                tags_list.append(tag.text)

            # 接下来找作者链接，进去爬取里面的信息
            author_link = 'http://quotes.toscrape.com/' + node.find_all('span')[1].a['href']
            # 抛给author_parse进行处理
            yield response.follow(author_link, self.author_parse)
            # print('{0:<4}:{1:<20} said:{2:<20}\n{3}'.format(self.page_num, author, text, tags_list))
            item = Lab3Item(author=author, text=text, tags=tags_list)
            yield item

        print('=' * 80 + 'page:',self.page_num,'saved successfully!' + '=' * 80)
        # 下面爬取下一页的链接
        try:
            self.page_num += 1
            url = soup.find('li', {'class': 'next'}).a['href']
            if url:
                next_link = 'http://quotes.toscrape.com/' + url
                yield scrapy.Request(next_link, callback=self.parse)
        except Exception:
            print('所有页面信息爬取结束！！！')

    def author_parse(self, response, **kwargs):
        soup = BeautifulSoup(response.body, 'html.parser')
        author_name = soup.find_all('div', {'class': 'author-details'})[0].find('h3').text
        birthday = soup.find('span').text
        bio = soup.find('div', {'class': 'author-description'}).text
        # print('{}: {}\n{}\n{}\n'.format(self.page_num, author_name, birthday, bio))
        item = Lab3Item(name=author_name, birthday=birthday, bio=bio)
        yield item
