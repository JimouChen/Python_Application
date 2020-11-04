"""
# @Time    :  2020/11/4
# @Author  :  Jimou Chen
"""
if __name__ == '__main__':

    str1 = input()
    str2 = input()
    len1 = len(str1)
    len2 = len(str2)
    dp = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    print(dp[i][j])

'''
xzyzzyx
zxyyzxz
4
'''
