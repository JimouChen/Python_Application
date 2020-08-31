"""
# @Time    :  2020/8/31
# @Author  :  Jimou Chen
"""
import wordcloud
import jieba

w = wordcloud.WordCloud(width=1000, height=700, background_color='white', font_path='msyh.ttc')
with open(r'data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text_list = jieba.lcut(text)
text = ' '.join(text_list)  # 变成string
w.generate(text)

w.to_file('demo1.png')
