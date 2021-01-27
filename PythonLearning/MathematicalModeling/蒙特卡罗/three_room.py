import random


def play(change):
    prize = random.randint(0, 2)
    guess = random.randint(0, 2)
    if prize == guess:
        if change:
            return False
        else:
            return True
    else:
        if change:
            return True
        else:
            return False


def winRate(change, N):
    win = 0
    for i in range(0, N):
        if play(change):
            win = win + 1
            # print('中奖率为: ')
            # print(win / N)
    print('中奖率为: ')
    print(win / N)


if __name__ == '__main__':
    N = 100000
    print('玩' + str(N) + '次，每一次都换门:')
    winRate(True, N)
    print()
    print('玩' + str(N) + '次，每一次都不换门:')
    winRate(False, N)
