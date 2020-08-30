# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class TestscrapyItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    page = scrapy.Field()
    name = scrapy.Field()
    word = scrapy.Field()
    tags = scrapy.Field()

    page_num = scrapy.Field()
    comment = scrapy.Field()

    author = scrapy.Field()
    birthday = scrapy.Field()
    bio = scrapy.Field()
