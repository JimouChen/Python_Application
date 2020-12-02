n = int(input())  # 几个作业
t = [[0 for _ in range(n)] for _ in range(2)]  # 2行代表2个机器的处理时间
now_work = [0 for _ in range(n)]  # 当前情况
best_work = [0 for _ in range(n)]  # 最优情况
now_t = 0
best_t = 9999999
# 机器1，2完成的时间
f1 = 0
f2 = 0
flag = [0 for _ in range(n)]  # 记录走过哪些点


def dfs(depth):
    global best_t, now_t, n, t, best_work, now_work, f1, f2, flag
    if depth >= n:
        if now_t < best_t:
            best_t = now_t
            for i in range(n):
                best_work[i] = now_work[i]
        return

    for i in range(n):
        if flag[i] == 0:
            f1 += t[0][i]
            last_f2 = f2
            if f1 < f2:
                f2 += t[1][i]
            else:
                f2 = f1 + t[1][i]

            # 减枝
            if f2 + now_t > best_t:
                f1 -= t[0][i]
                f2 = last_f2
                continue

            flag[i] = 1
            now_work[depth] = i
            now_t += f2

            dfs(depth + 1)
            now_t -= f2
            f1 -= t[0][i]
            f2 = last_f2
            flag[i] = 0


if __name__ == '__main__':
    # 输入两个机器对对应作业的处理时间
    for i in range(2):
        t[i] = list(map(int, input().split()))
    dfs(0)
    print(best_t)
    print(best_work)
    print(sum(t[0]) + t[1][-1])

'''
3
2 3 2
1 1 3
'''
