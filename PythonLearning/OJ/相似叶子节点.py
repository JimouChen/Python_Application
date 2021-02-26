# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:

    def travel(self, root: TreeNode, res: list):
        if not root:
            return
        if not root.right and not root.left:
            res.append(root.val)
            return
        self.travel(root.left, res)
        self.travel(root.right, res)

    def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
        res1 = []
        res2 = []
        self.travel(root1, res1)
        self.travel(root2, res2)

        return res2 == res1


a = Solution()
t1 = TreeNode(val=1)
t2 = TreeNode(val=1)
print(a.leafSimilar(t1, t2))
