def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


n = int(input())

while n:
    a0, a1, b0, b1 = map(int, input().split())
    x_num = 0
    p = a0 / a1
    q = b1 / b0

    x = 1
    while x * x <= b1:
        if b1 % x == 0:
            if x % a1 == 0 and gcd(x / a1, p) == 1 and gcd(q, b1 / x) == 1:
                x_num += 1
            y = b1 / x
            if x == y:
                continue
            if y % a1 == 0 and gcd(y / a1, p) == 1 and gcd(q, b1 / y) == 1:
                x_num += 1

        x += 1

    print(x_num)

    n -= 1
