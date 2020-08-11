"""
# @Time    :  2020/8/1
# @Author  :  Jimou Chen
"""


class Iterator:

    def __iter__(self):
        print('__iter__ was called')
        self.a = 1
        return self

    def __next__(self):
        if self.a < 10:
            x = self.a
            self.a += 1
            return x
        else:
            raise StopIteration


c = Iterator()
print(type(c))
for i in c:
    print(i)

it = iter(c)
print(type(it))

for i in it:
    print(i)
