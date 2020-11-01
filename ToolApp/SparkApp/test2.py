from pyspark import SparkContext, SparkConf

logFile = "file:////home/jmchen/PycharmProjects/python-application/ToolApp/SparkApp/test.txt"
conf = SparkConf().setMaster('local').setAppName('MyApp')

sc = SparkContext(conf=conf)
logData = sc.textFile(logFile, 2).cache()
numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()
print("Line with a:%i,lines with b :%i" % (numAs, numBs))
