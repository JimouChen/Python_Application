"""
# @Time    :  2020/10/25
# @Author  :  Jimou Chen
"""
orig = [[1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0]]

kernel = [[1, 0, 1],
          [0, 1, 0],
          [1, 0, 1]]


# 卷积计算
def cal_cov(N, n, step):
    new = [[0 for _ in range(N - n + step)] for _ in range(N - n + step)]
    for i in range(N - n + step):
        for j in range(N - n + step):
            temp = 0
            for k in range(n):
                for m in range(n):
                    temp += orig[k + i][m + j] * kernel[k][m]
            new[i][j] = temp
    return new


if __name__ == '__main__':
    print(cal_cov(5, 3, 1))
