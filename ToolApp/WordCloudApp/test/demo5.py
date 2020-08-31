"""
# @Time    :  2020/8/31
# @Author  :  Jimou Chen
"""
import jieba
import wordcloud
import imageio
import matplotlib.pyplot as plt
import itchat

'''
获取微信好友签名词云
'''
# 跳出二维码，登录微信
itchat.login()
# 获取好友列表
sign_list = []
friends_list = itchat.get_friends(update=True)
print('好友列表\n', friends_list)

print('每个人的签名:\n')
# 获取每个好友的签名
for each in friends_list:
    sign = each['Signature']
    print(sign)
    # 不要表情符号
    if 'emoji' in sign:
        pass
    else:
        sign_list.append(sign)

sign_string = ' '.join(sign_list)


# 读入词云的现状图片作为模板图片，一定要求白底的背景
shape = imageio.imread('china_map.png')
w = wordcloud.WordCloud(width=1000,
                        height=700,
                        background_color='white',
                        font_path='msyh.ttc',  # font_path是为了能够显示中文
                        mask=shape,  # 词云形状
                        # stopwords={'却说'},  # 可以加入不希望出现的词
                        # contour_width=10,  # 轮廓宽度
                        # contour_color='red',  # 轮廓颜色
                        scale=15)  # 清晰度，越大越好

text_list = jieba.lcut(sign_string, cut_all=True)
text = ' '.join(text_list)  # 变成string

w.generate(text)
# w.to_file('demo41.png')

# 显示图片
plt.imshow(w)
plt.axis('off')
plt.show()