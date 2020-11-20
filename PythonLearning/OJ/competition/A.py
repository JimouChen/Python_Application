import math


def f(n):
    if n == 1:
        return 1
    sum1 = 0
    for i in range(1, n):
        sum1 += (math.gcd(i, n - i) == 1)
    return sum1 % 1000000007


def g(n):
    sum1 = 0
    for i in range(1, n + 1):
        if n % i == 0:
            sum1 += f(n // i)

    return sum1 % 1000000007


def G(n, k):
    if k == 1:
        return f(g(n)) % 1000000007
    elif k > 1 and k % 2 == 0:
        return g(G(n, k - 1)) % 1000000007
    elif k > 1 and k % 2 == 1:
        return f(G(n, k - 1)) % 1000000007


t = int(input())
a = 1
while t:
    n, k = map(int, input().split())
    print(G(n % 1000000007, k % 1000000007))
    t -= 1
