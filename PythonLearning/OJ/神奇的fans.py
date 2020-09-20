"""
# @Time    :  2020/9/20
# @Author  :  Jimou Chen
"""
n = int(input())

for c in range(n):
    l_ = list(map(int, input().split()))
    if l_[0] > 2:

        l_.remove(l_[0])
        l_.sort()
        the = l_[2] - l_[1]
        flag = 0
        for i in range(1, len(l_)-1):
            if l_[i+1] - l_[i] == the:
                continue
            else:
                flag = 1
                break
        if flag == 1:
            print('no')
        else:
            print('yes')
    else:
        print('yes')

