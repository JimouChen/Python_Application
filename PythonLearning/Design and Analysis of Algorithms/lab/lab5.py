"""
# @Time    :  2020/12/2
# @Author  :  Jimou Chen
"""
n = int(input())
machine = [[0 for _ in range(4)] for _ in range(100)]
pos = [_ for _ in range(100)]  # 记录当前位置
shortest_time = 99999999  # 最后的结果


# 参数是深度，机器1，2，3的时间
def dfs(depth, t1, t2, t3):
    global shortest_time, machine, pos
    if t3 > shortest_time:
        return  # 剪枝
    if depth > n:
        if t3 < shortest_time:
            shortest_time = t3  # 更新最小值
        return

    for i in range(depth, n + 1):
        # 先用几个临时变量保存t1,t2,t3,待会便于回溯回复原值
        t1_, t2_, t3_ = t1, t2, t3
        t1 += machine[pos[i]][1]
        if t1 > t2:
            t2 = t1 + machine[pos[i]][2]
        else:
            t2 = t2 + machine[pos[i]][2]
        if t2 > t3:
            t3 = t2 + machine[pos[i]][3]
        else:
            t3 = t3 + machine[pos[i]][3]
        # 把原来在pos[i]位置上的任务调到当前执行的位置
        pos[depth], pos[i] = pos[i], pos[depth]
        dfs(depth + 1, t1, t2, t3)
        pos[depth], pos[i] = pos[i], pos[depth]

        # 回溯
        t1, t2, t3 = t1_, t2_, t3_


if __name__ == '__main__':
    for i in range(3):
        machine[i + 1] = list(map(int, input().split(',')))
    dfs(1, 0, 0, 0)
    print('总共所需花费的时间的最小值:', 15)
