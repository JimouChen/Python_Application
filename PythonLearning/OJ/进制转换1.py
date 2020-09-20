"""
# @Time    :  2020/9/20
# @Author  :  Jimou Chen
"""
while True:
    try:
        s = int(input())
        s = bin(s)[2:]
        print(s)
    except:
        break