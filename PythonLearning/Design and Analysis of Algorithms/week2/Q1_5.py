"""
# @Time    :  2020/9/17
# @Author  :  Jimou Chen
"""
num = '1111'
if __name__ == '__main__':

    while True:
        num += '1'
        if int(num) % 2013 == 0:
            print('至少需要{}个1'.format(len(num)))
            break


