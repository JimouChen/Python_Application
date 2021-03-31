n = int(input())
nums = list(map(int, input().split()))
res = nums[0]
for i in range(1, len(nums)):
    res = res ^ nums[i]

print(res)