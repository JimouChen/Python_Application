y, m, d = map(int, input().split())

r = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
p = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

if y % 4 == 0 and y % 100 != 0:
    s = 0
    for i in range(m - 1):
        s += r[i]
    s += d
    print(s)

else:
    s = 0
    for i in range(m - 1):
        s += p[i]
    s += d
    print(s)
