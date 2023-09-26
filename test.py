# import numpy as np
#
# ### 输入
# n, m = 3, 12
# t_list = [3, 4, 7]  # 生长时间
# a_list = [9, 3, 2]  # 买入价格
# b_list = [11, 6, 11]  # 卖出价格
# # n, m = map(lambda i: int(i), input().split(' '))
# # t_list = list(map(lambda i: int(i), input().split(' ')))
# # a_list = list(map(lambda i: int(i), input().split(' ')))
# # b_list = list(map(lambda i: int(i), input().split(' ')))
#
# t_list, a_list, b_list = map(lambda i: np.array(i), [t_list, a_list, b_list])
#
# v_max = 0
# v_list = [v_max]
#
# for i in range(1, m + 1):
#     # v_max = v_list[i - 1]
#     if i == 6:
#         print(b_list[0] - a_list[0], i - t_list[0], v_list[i - t_list[0]])
#     for index in range(n):
#         if i < t_list[index]:
#             continue
#         v_max = max(b_list[index] - a_list[index] + v_list[i - t_list[index]], v_max)
#     v_list.append(v_max)
#     print(i, v_max)
# print(len(v_list), v_list)
# print(v_max)
#
# str__ = '101110110'
#
#
# def str_num(str_, char_):
#     cnt = 0
#     for char_i in str_:
#         if char_i == char_:
#             cnt += 1
#     return cnt
#
#
# length = len(str__)
# v_arr = np.zeros((length + 1, length + 1))
# # v_arr[0, 0] = str_num(str__, '0')
# min_v = np.inf
# for i in range(1, length + 1):
#     v_arr[i, 0] = str_num(str__[:i], '1')  # 行表示前端
#     v_arr[0, i] = str_num(str__[-i:], '1')  # 列表是后端
#     for j in range(1, i + 1):
#         if (i + j) > length:
#             break
#         v_arr[i, j] = str_num(str__[i:-j], '0') + v_arr[0, j] + v_arr[i, 0]
#         v_arr[j, i] = str_num(str__[j:-i], '0') + v_arr[0, i] + v_arr[j, 0]
#         if min_v > v_arr[i, j]:
#             min_v = v_arr[i, j]
#         if min_v > v_arr[j, i]:
#             min_v = v_arr[j, i]
#
# for i in range(length + 1):
#     v_arr[i, 0] += str_num(str__[i:], '0')
#     v_arr[0, i] += str_num(str__[:-i], '0')
#     if min_v > v_arr[i, 0]:
#         min_v = v_arr[i, 0]
#     if min_v > v_arr[0, i]:
#         min_v = v_arr[0, i]
# print(v_arr)
# print(min_v)


# class Solution(object):
#     def twoSum(self, nums, target):
#         """
#         :type nums: List[int]
#         :type target: int
#         :rtype: List[int]
#         """
#         import numpy as np
#         nums_ori = nums
#         nums = np.array(nums)
#         indexs = sorted(range(len(nums)), key=lambda index: nums[index])
#         # res = list(sorted(zip(range(len(nums)), nums), key=lambda k: k[1]))
#         print(indexs)
#         nums = nums[indexs]
#         # indexs = res[:, 0]
#         # nums = res[:, 1]
#
#         # print(indexs)
#         # # return indexs
#         # # nums = sorted(nums)
#         # print(indexs)
#         # nums = nums[indexs]
#         # return nums
#         start_index = 0
#         end_index = nums.shape[0]
#         # while(nums[end_index] > target):
#         #     end_index-=1
#         while (True):
#             num_1 = nums[start_index]
#             num_2 = nums[end_index]
#             sum = num_1 + num_2
#             if sum == target:
#                 break
#             if sum > target:
#                 end_index -= 1
#             if sum < target:
#                 start_index += 1
#
#         return indexs[start_index], indexs[end_index]
#
#
# nums = [2, 7, 11, 9]
# target = 9
# solution = Solution()
# print(solution.twoSum(nums, target))

def harmonic_mean(x, y):
    return (x * y) / (x + y)


def square_mean(x, y):
    return np.sqrt((x ** 2 + y ** 2) / 2)


def square_mean_2(x, y, n):
    return np.power((np.power(x, n) + np.power(y, n)) / 2, 1/n)


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

number = 10
x = np.linspace(1, 20, number)
y = np.linspace(1, 20, number)
shape = (10, 10)
# a = np.linalg.norm(x[0] - y[0], 1)
heat_map = np.zeros((number, number))
for i in range(number):
    for j in range(number):
        # data_A = np.random.normal(loc=x[i], scale=1, size=shape)
        # data_B = np.random.normal(loc=y[j], scale=1, size=shape)

        # heat_map[i][j] = -np.abs((x[i] - y[j]) / aaa(x[i],y[j]))
        # heat_map[i][j] = -np.abs((x[i] - y[j]) / y[j])
        # heat_map[i][j] = -np.abs((x[i] - y[j]))
        # heat_map[i][j] = -np.abs((x[i] - y[j])) / square_mean(x[i],y[j])
        heat_map[i][j] = -np.abs((x[i] - y[j])) / square_mean_2(x[i], y[j], 10)
        # heat_map[i][j] = -np.abs((x[i] - y[j])) / (x[i]+y[j])
        # heat_map[i][j] = np.linalg.norm((data_A-data_B)/data_A,1)/data_A.size
        # heat_map[i][j] = np.linalg.norm((data_A-data_B)/data_B,1)/data_A.size
        # heat_map[i][j] = np.linalg.norm((data_A - data_B) / aaa(data_A, data_B), 1) / data_A.size
plt.rcParams['font.sans-serif'] = ['SimHei']
# 坐标轴负号的处理
plt.rcParams['axes.unicode_minus'] = False
# 设置下三角mask遮罩，上三角将i,j互换即可
sns.heatmap(data=heat_map, cmap='RdBu', vmax=0, vmin=-2, center=-1, annot=True, square=True, linewidths=0,
            cbar_kws={"shrink": .6}, xticklabels=True, yticklabels=True, fmt='.2f')
# center 值越大颜色越浅
# shrink:n n为缩短的比例(0-1)
# fmt='.2f' 显示数字保留两位小数
plt.title('热力图', fontsize='xx-large', fontweight='heavy')
# 设置标题字体
plt.show()
