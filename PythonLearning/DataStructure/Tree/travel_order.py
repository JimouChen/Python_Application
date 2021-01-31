class TreeNode:
    def __init__(self, left=None, right=None, data=0):
        self.data = data
        self.left = left
        self.right = right


def pre_travel(root: TreeNode):
    if root:
        print(root.data, end=' ')
        pre_travel(root.left)
        pre_travel(root.right)


def mid_travel(root: TreeNode):
    if root:
        mid_travel(root.left)
        print(root.data, end=' ')
        mid_travel(root.right)


def after_travel(root: TreeNode):
    if root:
        after_travel(root.left)
        after_travel(root.right)
        print(root.data, end=' ')


if __name__ == '__main__':
    node1 = TreeNode(data=1)
    node2 = TreeNode(data=2)
    node3 = TreeNode(data=3)
    node4 = TreeNode(data=4)
    node5 = TreeNode(data=5)

    '''
       1
     2   3
    4 5
    '''
    node1.left = node2
    node1.right = node3
    node2.left = node4
    node2.right = node5
    print('前序遍历: ', end='')
    pre_travel(node1)
    print('\n中序遍历: ', end='')
    mid_travel(node1)
    print('\n后序遍历: ', end='')
    after_travel(node1)
