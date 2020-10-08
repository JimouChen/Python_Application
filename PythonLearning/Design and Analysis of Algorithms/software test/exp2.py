def judge(a, b, c):
    if (a < -200 or a > 200) and (b < -200 or b > 200) and (c < -200 or c > 200):
        print('A、B、C不在有效值域-200至200中')
    elif (a < -200 or a > 200) and (b < -200 or b > 200):
        print('A、B不在有效值域-200至200中')
    elif (a < -200 or a > 200) and (c < -200 or c > 200):
        print('A、C不在有效值域-200至200中')
    elif (b < -200 or b > 200) and (c < -200 or c > 200):
        print('B、C不在有效值域-200至200中')
    elif a < -200 or a > 200:
        print('A不在有效值域-200至200中')
    elif b < -200 or b > 200:
        print('B不在有效值域-200至200中')
    elif c < -200 or c > 200:
        print('C不在有效值域-200至200中')


while True:
    try:
        a, b, c = map(int, input('请输入A、B、C的值:').split())
        if a > 200 or a < -200 or b > 200 or b < -200 or c > 200 or c < -200:
            print('输入错误!', end='')
            judge(a, b, c)
        else:
            if a == 0:
                if b == 0:
                    if c == 0:
                        print('方程AX^2 + BX + C = 0有无数个根')
                    else:
                        print('方程AX^2 + BX + C = 0 无根')
                else:
                    x = -c / b
                    if x == 0:
                        x = 0
                    print('方程AX^2 + BX + C = 0有1个实根, x = %.2f' % x)
            else:
                # a 不等于0的情况
                delta = b ** 2 - 4 * a * c
                if delta > 0:
                    x1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
                    x2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
                    print('方程AX^2 + BX + C = 0有2个不同的实根, x1 = %.2f，x2 = %.2f' % (x1, x2))

                elif delta == 0:
                    x = -b / (2 * a)
                    if x == 0:
                        x = 0
                    print('方程AX^2 + BX + C = 0有2个相同的实根, x1 = x2 = %.2f' % x)
                elif delta < 0:
                    x1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
                    x2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
                    x1_real = round(x1.real, 2)
                    x2_real = round(x2.real, 2)
                    x_img = round(abs(x1.imag), 2)
                    if x1.imag < 0:
                        x1_imag = str(round(x1.imag))
                    else:
                        x1_imag = '+' + str(round(x1.imag))

                    print(
                        '方程AX^2 + BX + C = 0有2个不同的虚根, x1 = {}+{}j ，x2 = {}-{}j'.format(x1_real, x_img, x2_real, x_img))
    except:
        break
