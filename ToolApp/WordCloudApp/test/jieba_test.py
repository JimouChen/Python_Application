"""
# @Time    :  2020/8/31
# @Author  :  Jimou Chen
"""
import jieba

text_list = jieba.lcut('粉丝的芳草飞机饿哦平均分')
print(text_list)
text = '---'.join(text_list)
print(text)