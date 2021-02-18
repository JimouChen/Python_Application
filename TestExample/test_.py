# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# import seaborn  # 画热力图
#
#
# def handle_data():
#     data = pd.read_csv('data/new_data.csv')
#     le = LabelEncoder()
#     data.iloc[:, 0] = le.fit_transform(data.iloc[:, 0])
#
#     return data
#
#
# def draw_heat_map(the_data):
#     # 画热力图
#     plt.figure(figsize=(15, 15))
#     p = seaborn.heatmap(the_data.corr(), annot=True, annot_kws={'fontsize': 20}, square=True, cmap="YlGnBu")
#     # 设置X轴标签的字体大小和字体颜色
#     # p.set_xlabel('X Label', fontsize=30)
#     # 设置Y轴标签的字体大小和字体颜色
#     # p.set_ylabel('Y Label', fontsize=30, color='r')
#     plt.tick_params(labelsize=15)  # x,y轴刻度上的字体大小
#     p.set_yticklabels(p.get_yticklabels(), rotation=45)
#     plt.show()
#
#
# if __name__ == '__main__':
#     data = handle_data()
#     x_data = data.iloc[:, 1:]
#     y_data = data.iloc[:, 0]
#     draw_heat_map(data)
#     # data.to_csv('data/new_data.csv')
#     # new_data = pd.read_csv('data/new_data.csv')
#     # print(new_data)

z = [0.60553896,-0.22906616,1.86852386]
print(sum(z))