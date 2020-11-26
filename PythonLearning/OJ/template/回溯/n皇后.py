"""
# @Time    :  2020/11/26
# @Author  :  Jimou Chen
"""
n = int(input())
chess = [0 for _ in range(n)]  # 下标代表第i行，若为1，则表示第j列棋盘存放棋子
cnt = 0


# 检查是否冲突，i是行，j是列
def check(i):
    for j in range(i):
        # 检查列和对角线
        if chess[i] == chess[j] or abs(chess[i] - chess[j]) == i - j:
            return 0
    return 1


# i表示现在已经放到第i行了
def dfs(i):
    global cnt
    if i == n:  # 能够放到最后一行说明这种情况符合
        cnt += 1
        print(chess)
        return

    for j in range(n):
        chess[i] = j  # 表示第i行第j列放皇后
        if check(i):
            dfs(i + 1)  # 符合条件就继续放下一行


if __name__ == '__main__':
    dfs(0)
    print('一共有{}种摆放方法'.format(cnt))
