from pymprog import *

if __name__ == '__main__':
    begin('bike production')
    x, y = var('x, y')  # variables
    maximize(15 * x + 10 * y, 'profit')
    x <= 3  # mountain bike limit
    y <= 4  # racer production limit
    x + y <= 5  # metal finishing limit
    solve()

    print('x取值：' + str(x.primal))
    print('y取值：' + str(y.primal))
    print('最优解为：' + str(vobj()))
