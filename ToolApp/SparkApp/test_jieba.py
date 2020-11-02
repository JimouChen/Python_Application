import jieba
from pyspark import SparkContext

if __name__ == '__main__':
    # load file
    with open('test.txt', 'r') as f:
        file = f.read()

    res_str = ''
    txt_str = ''
    clear_list = ['\n', '。', '，', '， ', '！', '“', '”', '…', ' ', "'", '?', ':','、']

    for i in file:
        if i not in clear_list:
            res_str += i

    res_list = list((jieba.cut(res_str, cut_all=False)))
    # print(res_list)

    sc = SparkContext('local', "WordCount")
    sc.setLogLevel("ERROR")

    rdd = sc.parallelize(res_list)
    res_dict = rdd.flatMap(lambda x: x.split(" ")).countByValue()
    res_dict = dict(sorted(res_dict.items(), key=lambda x: x[1], reverse=True))
    # print(res_dict)

    for key, value in res_dict.items():
        print("%s %i" % (key, value))
    # stop spark
    sc.stop()
