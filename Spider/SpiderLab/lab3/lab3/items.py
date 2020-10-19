# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class Lab3Item(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    author = scrapy.Field()
    text = scrapy.Field()
    tags = scrapy.Field()

    name = scrapy.Field()
    birthday = scrapy.Field()
    bio = scrapy.Field()
