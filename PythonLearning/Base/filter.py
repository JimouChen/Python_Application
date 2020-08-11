"""
# @Time    :  2020/7/17
# @Author  :  Jimou Chen
"""

# 前面是None的话，过滤掉所有0和False
f = filter(None, [1, 2, 3, True, 0, False])
f = list(f)
print(f)

# 也可以自定义过滤条件
# 比如过滤偶数
f = list(filter(lambda x: x % 2, range(10)))
print(f)


# 或者传个函数名给他
def odd(x):
    return x % 2


print(list(filter(odd, range(10))))
