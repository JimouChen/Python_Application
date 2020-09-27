n = int(input())
s = 0
for i in range(1, n+1):
    if ('2' in str(i)) or ('0' in str(i)) or ('1' in str(i)) or ('9' in str(i)) and str(i)[0] != '0':
        s += i

print(s)