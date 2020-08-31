"""
# @Time    :  2020/8/31
# @Author  :  Jimou Chen
"""
import jieba
import wordcloud
import imageio

'''
有形状的词云
'''

# 读入词云的现状图片
shape = imageio.imread('star.png')

w = wordcloud.WordCloud(width=1000,
                        height=700,
                        background_color='white',
                        font_path='msyh.ttc',  # font_path是为了能够显示中文
                        mask=shape,  # 词云形状
                        scale=15)  # 清晰度，越大越好
# 读入词云内容
with open(r'data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text_list = jieba.lcut(text)
text = ' '.join(text_list)  # 变成string

w.generate(text)

w.to_file('demo3.png')
