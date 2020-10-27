"""
# @Time    :  2020/10/27
# @Author  :  Jimou Chen
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


a = ListNode(1)
b = ListNode(2)
c = ListNode(3)
a.next = b
b.next = c

l = a
while l:
    print(l.val)
    l = l.next
