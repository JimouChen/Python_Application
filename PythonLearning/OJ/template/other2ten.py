"""
# @Time    :  2020/11/1
# @Author  :  Jimou Chen
"""
'''16 -> 10'''
n = input()
print(int(n, 16))

'''8->10'''
print(int(n, 8))

'''2->10'''
print(int(n, 2))

'''10->2'''

# 获取用户输入十进制数
dec = int(input("输入数字："))

print("十进制数为：", dec)
print("转换为二进制为：", bin(dec))
print("转换为八进制为：", oct(dec))
print("转换为八进制为：%o" % dec)
print("转换为十六进制为：", hex(dec))
print("转换为十六进制为：%X" % dec)
print("转换为十六进制为：%x" % dec)

'''
https://www.cnblogs.com/aaronthon/p/9446048.html
'''
