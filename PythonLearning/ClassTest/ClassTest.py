"""
# @Time    :  2020/6/18
# @Author  :  Jimou Chen
"""


class People(object):
    count = 0

    def __init__(self, name):
        self.name = name
        People.count += 1

    # def setName(self, name):
    #     self.name = name

    def speak(self):
        print("hello world!")


class Child(People):
    def __init__(self, name):
        super().__init__(name)
        print("sdsf")

    def speak(self):
        print("yesyseyes")


if __name__ == '__main__':
    Tom = People("tt")
    Peter = People("ww")
    Tom.speak()

    print(Tom.name)
    print(People.count + Tom.count)
    BB = Child("ray")
    print(BB.name + " say ")
