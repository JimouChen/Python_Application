"""
# @Time    :  2020/6/24
# @Author  :  Jimou Chen
"""


class Stack:
    def __init__(self):
        self.elem = []

    def pop(self):
        self.elem.pop()

    def push(self, obj):
        self.elem.append(obj)

    def get_pop(self):
        return self.elem[-1]

    def is_empty(self):
        if len(self.elem) == 0:
            return True
        else:
            return False

    def length(self):
        return len(self.elem)

    def show(self):
        print(self.elem)


if __name__ == '__main__':
    stack = Stack()
    stack.elem = [1, 2, 3, 4, 5, 6, 7]
    stack.show()
    stack.pop()
    stack.show()
    stack.push(999)
    stack.show()
    if stack.is_empty():
        print("empty")
    else:
        print("no empty")
