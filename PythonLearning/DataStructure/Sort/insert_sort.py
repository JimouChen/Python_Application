def insertSort(array: list):
    for i in range(1, len(array)):
        if array[i] < array[i - 1]:
            temp = array[i]
            j = i - 1
            # 如果还有比该元素大的元素，且前面还没全部比完
            while j >= 0 and temp < array[j]:
                array[j + 1] = array[j]  # 前面的元素后移
                j -= 1

            array[j + 1] = temp

    return array


if __name__ == '__main__':
    test = [3, 2, 7, 6, 9, 9, 1, 0, 54, 23]
    # test = [2, 3, 1, 0, 54, 23]
    print(insertSort(test))
