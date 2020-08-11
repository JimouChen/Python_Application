"""
# @Time    :  2020/7/17
# @Author  :  Jimou Chen
"""
'''
进制转换
'''
a = 108
print('%#o' % a)
print('%#x' % a)

# 或者下面这种方式

# 获取用户输入十进制数
dec = int(input("输入数字："))

print("十进制数为：", dec)
print("转换为二进制为：", bin(dec))
print("转换为八进制为：", oct(dec))
print("转换为十六进制为：", hex(dec))
