"""
# @Time    :  2020/7/17
# @Author  :  Jimou Chen
"""
'''
玛雅人有一种密码，如果字符串中出现连续的2012四个数字就能解开密码。给个长度为N的字符串，（2=N<=13）该字符串中只含有0,1,2三种数字，
问这个字符串要移位几次才能解开密码，每次只能移动相邻的两个数字。例如02120经过一次移位，可以得到20120,01220,02210,02102，
其中20120符合要求，因此输出为1如果无论移位多少次都解不开密码，输出-1
'''


def judge(s):
    if '2012' in s:
        return 1
    else:
        return 0


def list_to_str(l):
    a = ''
    for i in l:
        a += i
    return a


flag = [0]
while True:
    try:
        c = 0
        flag[0] = 0
        n = input()
        str1 = input()
        list_s = list(str1)

        for i in range(len(list_s) - 1):
            if judge(list_to_str(list_s)):
                print(1)
                flag[0] = 1
                break
            else:
                list_s[i], list_s[i + 1] = list_s[i + 1], list_s[i]
                if judge(list_to_str(list_s)):
                    print(1)
                    flag[0] = 1
                    break
                # print(list_to_str(list_s))
                list_s[i], list_s[i + 1] = list_s[i + 1], list_s[i]

        if flag[0] == 0:
            print(-1)

    except:
        break

#
# s = '123'
# print(s)
# s = list(s)
# print(s)
# a = ''
# for i in s:
#     a += i
# print(a)
