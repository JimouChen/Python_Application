# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:

        p = head
        all_val = []
        while p:
            q = ListNode(0)
            q.val = p.val
            all_val.append(q)
            p = p.next

        all_val.remove(all_val[len(all_val) - n])
        if len(all_val) == 0:
            return None
        for i in range(len(all_val) - 1):
            all_val[i].next = all_val[i + 1]

        return all_val[0]


a = Solution()
t1 = ListNode(1)
t2 = ListNode(2)
# t3 = ListNode(3)
# t4 = ListNode(4)
# t5 = ListNode(5)
t1.next = t2
# t2.next = t3
# t3.next = t4
# t4.next = t5

a.removeNthFromEnd(t1, 1)
