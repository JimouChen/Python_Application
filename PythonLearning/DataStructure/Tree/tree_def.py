"""
# @Time    :  2020/9/30
# @Author  :  Jimou Chen
"""


# 定义二叉树节点
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 先序创建二叉树
def createTree(t: TreeNode):
    ch = input()
    if ch == '#':
        t = None
    else:
        t = TreeNode(val=ch)
        t.left = createTree(t.left)
        t.right = createTree(t.right)
    return t


# 先序遍历
def preOrder(t: TreeNode):
    if t:
        print(t.val, end=' ')
        preOrder(t.left)
        preOrder(t.right)


# 中序
def inOrder(t: TreeNode):
    if t:
        inOrder(t.left)
        print(t.val, end=' ')
        inOrder(t.right)


# 后序
def postOrder(t: TreeNode):
    if t:
        postOrder(t.left)
        postOrder(t.right)
        print(t.val, end=' ')


# 获取叶子节点
leavesNode = []


def getLeaveNode(t: TreeNode):
    if not t.left and not t.right:
        leavesNode.append(t.val)
        return
    if t.left:
        getLeaveNode(t.left)
    if t.right:
        getLeaveNode(t.right)


if __name__ == '__main__':
    tree = TreeNode()
    tree = createTree(tree)
    preOrder(tree)
    print()
    inOrder(tree)
    print()
    postOrder(tree)
    print('\n over!')
    getLeaveNode(tree)
    print(leavesNode)

'''
a
b
d
#
g
j
#
#
m
#
#
#
c
e
#
#
f
h
#
#
i
#
#
'''
