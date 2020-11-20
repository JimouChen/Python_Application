l = [0 for i in range(30)]
l[0] = 1
s = l[0]
for i in range(1, 30):
    l[i] = (l[i - 1] * (l[i - 1] + 5)) % 1000000007
    s += l[i]
    # print(s)

print(s % 1000000007)
print('937527335', end='')
