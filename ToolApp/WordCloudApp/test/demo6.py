"""
# @Time    :  2020/8/31
# @Author  :  Jimou Chen
"""
import snownlp

print('\n')
text1 = '中华民族伟大复兴'
print('{:-^50}'.format('测试文本：' + text1))
s = snownlp.SnowNLP(text1)
print(s, '\n')
print('情感分析', s.sentiments)
print('\n')
print('中文分词', s.words)
print('\n')
print('转成拼音', s.pinyin)
print('\n')
print('词频', s.tf)
print('\n')
print('提取三个关键词', s.keywords(3))
print('\n')

text2 = '快递慢到死，你们客服态度不好，你快快退款！'
print('{:-^50}'.format('测试文本：' + text2))
s = snownlp.SnowNLP(text2)
print('\n')
print('情感分析', s.sentiments)
print('\n')
print('中文分词', s.words)
print('\n')
print('转成拼音', s.pinyin)
print('\n')
print('词频', s.tf)
print('\n')
print('提取三个关键词', s.keywords(3))