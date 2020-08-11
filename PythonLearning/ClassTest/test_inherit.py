"""
# @Time    :  2020/7/17
# @Author  :  Jimou Chen
"""


class Fish:
    def __init__(self):
        print('123456')


class Shark(Fish):
    def __init__(self, name):
        self.name = name
        super().__init__()

    def move(self, time):
        self.time = time
        print(self.name + '  is swimming for ' + str(self.time))


shark = Shark('skkk')
shark.move(5)
