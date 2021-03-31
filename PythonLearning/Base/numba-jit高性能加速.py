from numba import jit


@jit()
def test(n):
    a = 0
    for i in range(n):
        a += i
    return a


def no_jit(n):
    a = 0
    for i in range(n):
        a += i
    return a


if __name__ == '__main__':
    print(test(1000000000))
    print(no_jit(1000000000))
