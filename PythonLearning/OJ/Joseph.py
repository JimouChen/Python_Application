"""
约瑟夫问题的多种解法
n个人,数到m就去掉
"""


# 数组法
def joseph_by_array(n, m):
    a = [0 for _ in range(n + 1)]
    cnt = 0  # 目前出局的人数
    i, k = 0, 0  # a[i]是报数的人，k是报的数，从1开始数
    while cnt != n:
        i += 1
        if i > n:
            i = 1
        if a[i] == 0:
            k += 1
            if k == m:
                a[i] = 1
                cnt += 1
                print(i, end=" ")
                k = 0  # 重新从1开始报数


# 索引法remove_index = (remove_index + m - 1) % len(number)
def joseph_by_index(n, m):
    number = [_ for _ in range(1, n + 1)]
    out_index = 0
    while len(number):
        out_index = (out_index + m - 1) % len(number)
        print(number[out_index], end=" ")
        number.remove(number[out_index])


# 公式法公式法更快：f(n, m) = (f(n - 1, m) + m) mod n


if __name__ == '__main__':
    person_num, out_num = map(int, input().split())
    joseph_by_array(person_num, out_num)
    print()
    joseph_by_index(person_num, out_num)
