while True:
    try:
        s = input()

        if s == 'E':
            break
        j_num = s.count('J')
        z_num = s.count('Z')
        o_num = s.count('O')

        z = list('Z' * z_num)
        o = list('O' * o_num)
        j = list('J' * j_num)

        while len(z) or len(o) or len(j):
            if len(z):
                print(z.pop(), end='')
            if len(o):
                print(o.pop(), end='')
            if len(j):
                print(j.pop(), end='')

        print()

    except:
        break
