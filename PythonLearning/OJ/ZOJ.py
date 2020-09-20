"""
# @Time    :  2020/9/20
# @Author  :  Jimou Chen
"""
# ZOJZOJOZJOZJOZJZOJZOJZOJZOJZ
while True:
    try:
        s = list(input())
        if s[0] == 'E':
            break
        s.sort()
        s.reverse()
        z = []
        o = []
        j = []
        for i in s:
            if i == 'Z':
                z.append(i)
            if i == 'O':
                o.append(i)
            if i == 'J':
                j.append(i)
        print(z, o, j)
        for i in z:
            if len(z) != 0:
                print(i, end='')
                z.pop(0)
            for p in o:
                if len(o) != 0:
                    print(p, end='')
                    j.pop(0)
                for k in j:
                    if len(j) != 0:
                        print(k, end='')
                        j.pop(0)


        # res = []
        # for i in s:
        #     if i == 'Z':
        #         res.append(i)
        #         s.remove()
    except:
        break
