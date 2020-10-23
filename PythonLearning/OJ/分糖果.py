n = int(input())
kids = list(map(int, input().split()))
c = 0
while len(set(kids)) != 1:
    temp = kids[0]
    for i in range(len(kids) - 1):
        kids[i] -= kids[i] >> 1
        kids[i] += kids[i + 1] >> 1
        if kids[i] & 1 == 1:
            c += 1
            kids[i] += 1

    kids[n - 1] -= kids[n - 1] >> 1
    kids[n - 1] += temp >> 1
    if kids[n - 1] & 1 == 1:
        c += 1
        kids[n - 1] += 1

print(c)
