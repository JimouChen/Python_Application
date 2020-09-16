"""
# @Time    :  2020/9/17
# @Author  :  Jimou Chen
"""
num = '1111'
while True:
    num += '1'
    if int(num) % 2013 == 0:
        n = len(num)
        print(n)
        break

