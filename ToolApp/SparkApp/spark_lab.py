import jieba
from pyspark import SparkContext

if __name__ == '__main__':
    # 读入文件
    with open('hwzy.txt', 'r') as f:
        file = f.read()

    res_str = ''
    txt_str = ''
    clear_list = ['\n', '。', '，', '， ', '！', '“', '”', '…', ' ', "'", '?', '？', ':', '、', '：']
    # 去掉文本中的标点符号和其他非文字
    for i in file:
        if i not in clear_list:
            res_str += i

    res_list = list((jieba.cut(res_str, cut_all=False)))
    sc = SparkContext('local', "WordCount")
    sc.setLogLevel("ERROR")
    rdd = sc.parallelize(res_list)
    res_dict = rdd.flatMap(lambda x: x.split(" ")).countByValue()
    res_dict = dict(sorted(res_dict.items(), key=lambda x: x[1], reverse=True))

    with open('result.txt', 'w+')as save_file:
        for key, value in res_dict.items():
            save_file.write("%s %i\n" % (key, value))
            print("%s %i" % (key, value))
    # 停止 spark
    sc.stop()
