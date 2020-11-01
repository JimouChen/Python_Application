from pyspark import SparkContext, SparkConf


def split(x):
    words = x.split()
    return [(word, 1) for word in words]


if __name__ == '__main__':
    # set sparkcontext
    conf = SparkConf().setMaster("local[*]").setAppName("My App")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    rdd = sc.textFile('page10.txt')
    words = rdd.flatMap(split)
    count = words.reduceByKey(lambda x, y: x + y).sortBy(lambda x: x[1],
                                                         ascending=False)
    count.foreach(print)
    # stop spark
    sc.stop()
