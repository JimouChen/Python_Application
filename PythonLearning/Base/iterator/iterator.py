"""
# @Time    :  2020/7/18
# @Author  :  Jimou Chen
"""
string = 'hello world'
it = iter(string)

while True:
    try:
        each = next(it)
        print(each)
    except:
        break
