"""
# @Time    :  2020/10/29
# @Author  :  Jimou Chen
"""
def count_sort(array: list):
    max_val = max(array)
    cnt = [0 for _ in range(max_val + 1)]

    for i in array:
        cnt[i] += 1

    for i in range(1, len(cnt)):
        print(str(chr(i)) * cnt[i], end='')


if __name__ == '__main__':
    a = input()
    test = []
    for i in a:
        test.append(ord(i))

    count_sort(test)