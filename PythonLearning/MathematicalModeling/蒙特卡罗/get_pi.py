import math
import random

if __name__ == '__main__':

    M = input('请输入一个较大的整数')
    N = 0
    for i in range(int(M)):
        x = random.random()
        y = random.random()
        if math.sqrt(x ** 2 + y ** 2) < 1:
            N += 1
            pi = 4 * N / int(M)
            # print(pi)
    print(pi)
