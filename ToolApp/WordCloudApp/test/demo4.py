"""
# @Time    :  2020/8/31
# @Author  :  Jimou Chen
"""
import jieba
import wordcloud
import imageio
import matplotlib.pyplot as plt

'''
有形状,轮廓的词云
'''

# 读入词云的现状图片作为模板图片，一定要求白底的背景
shape = imageio.imread('queen.jpg')

w = wordcloud.WordCloud(width=1000,
                        height=700,
                        background_color='white',
                        font_path='msyh.ttc',  # font_path是为了能够显示中文
                        mask=shape,  # 词云形状
                        # stopwords={'却说'},  # 可以加入不希望出现的词
                        contour_width=20,  # 轮廓宽度
                        contour_color='red',  # 轮廓颜色
                        scale=15)  # 清晰度，越大越好
# 读入词云内容
with open(r'hamlet.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text_list = jieba.lcut(text)
text = ' '.join(text_list)  # 变成string

w.generate(text)

w.to_file('demo41.png')

# 显示图片
plt.imshow(w)
plt.axis('off')
plt.show()