"""
# @Time    :  2020/7/19
# @Author  :  Jimou Chen
"""
import re

p = re.compile(r'[A-Z]')
print(p.search('Hello World'))
print(p.findall('Hello World'))

# 不用compile的情况
print(re.search(r'[A-Z]', 'Hello World'))
print(re.findall(r'[A-Z]', 'Hello World'))

# 使用group
obj = re.search(r' (\w+) (\w+)', 'I am HaHa.')
print(obj.group())
