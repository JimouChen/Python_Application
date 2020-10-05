"""
# @Time    :  2020/10/2
# @Author  :  Jimou Chen
"""
import numpy
while True:

    try:

        a = list(map(int, input().split()))
        c = []
        count = numpy.bincount(a)
        # print(count)
        index = numpy.argmax(count)
        print(index)



    except:
        break

# 下面不使用库
'''
while True:
    try:

        a = list(map(int, input().split()))
        c = []
        a.sort()
        for i in a:
            c.append(a.count(i))
        max_index = c.index(max(c))
        print(a[max_index])

    except:
        break

'''