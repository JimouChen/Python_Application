"""
找到所有解空间是基本，然后做剪枝
以全排列为例子
"""
flag = [0 for _ in range(100)]


# 这是使用标记位来判断是否访问元素,最常用
def backtrack(nums: list, box: list):
    if len(nums) == len(box):
        print(box)
        return

    for i in range(len(nums)):
        if flag[i] == 1:
            continue
        box.append(nums[i])
        flag[i] = 1
        backtrack(nums, box)
        box.pop()
        flag[i] = 0


# 未使用标记位，最原始的思想，速度不如上面那种
def backtrack_org(nums: list, box: list):
    if len(nums) == len(box):
        print(box)
        return

    for i in range(len(nums)):
        if nums[i] in box:
            continue
        box.append(nums[i])
        backtrack_org(nums, box)
        box.pop()


if __name__ == '__main__':
    test = [1, 3, -1]
    backtrack(test, [])

    backtrack_org([1, 2, 3], [])
