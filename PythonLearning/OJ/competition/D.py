n = int(input())
dao = []
for i in range(4):
    l = input()
    dao.append(l)


for i in range(4):
    speed = 1

    for j in dao[i]:
        if dao[i] == '.':
            speed = 1
        elif dao[i] == 'w':
            speed = 0.5
        elif dao[i] == '>':
            speed *= 2

