# -*- coding: utf-8 -*-

import os
import pandas as pd



def three_zone_ratio(red_num):
    """
    功能：计算三区比、断区数
    参数：
        dataset：原始数据集
        red_num_columns：红球的列名
    返回值：三区比、断区数
    """
    three_zone_list = []
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        interval_1, interval_2, interval_3 = 0, 0, 0
        # 对 red_num 中的列进行循环
        for number in red_num.loc[i, :]:
            if 1 <= number <= 11:
                interval_1 = interval_1 + 1
            elif 12 <= number <= 22:
                interval_2 = interval_2 + 1
            elif 23 <= number <= 33:
                interval_3 = interval_3 + 1
        # 计算断区数
        number_of_breaks = [interval_1, interval_2, interval_3].count(0)
        # 把三区比和断区数添加到 three_zone_list 中
        three_zone_list.append([interval_1, interval_2, interval_3, number_of_breaks])
    three_zone = pd.DataFrame(three_zone_list, columns=["三区_1", "三区_2", "三区_3", "三区_断区数"])
    return three_zone



def four_district_ratio(red_num):
    """
    功能：计算四区比、断区数
    参数：
        dataset：原始数据集
        red_num_columns：红球的列名
    返回值：四区比、断区数
    """
    four_districts_list = []
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        interval_1, interval_2, interval_3, interval_4 = 0, 0, 0, 0
        # 对 red_num 中的列进行循环
        for number in red_num.loc[i, :]:
            if 1 <= number <= 8:
                interval_1 = interval_1 + 1
            elif 9 <= number <= 16:
                interval_2 = interval_2 + 1
            elif 18 <= number <= 25:
                interval_3 = interval_3 + 1
            elif 26 <= number <= 33:
                interval_4 = interval_4 + 1
        # 计算断区数
        number_of_breaks = [interval_1, interval_2, interval_3, interval_4].count(0)
        # 把四区比和断区数添加到 three_zone_list 中
        four_districts_list.append([interval_1, interval_2, interval_3, interval_4, number_of_breaks])
    four_districts = pd.DataFrame(four_districts_list, columns=["四区_1", "四区_2", "四区_3", "四区_4", "四区_断区数"])
    return four_districts



def interval_ratio(red_num):
    """
    功能：计算三区比、四区比、断区数
    参数：
        dataset：原始数据集
        red_num_columns：红球的列名
    返回值：三区比、四区比、断区数
    """
    # 三区比
    three_zone = three_zone_ratio(red_num)
    # 四区比
    four_districts = four_district_ratio(red_num)
    # 区间数据
    interval_data = pd.concat([three_zone, four_districts], axis=1)
    return interval_data



def with_value(red_num):
    """
    功能：计算红球总体和值、首尾和值
    参数：
        dataset：原始数据集
        red_num_columns：红球的列名
    返回值：和值、首尾和值
    """
    with_value = red_num.sum(axis=1)  # 和值
    red1_add_red6 = red_num["red1"] + red_num["red6"]  # 首尾和值
    # 分别把 with_value、red1_add_red6 转换为 DataFrame
    with_value = pd.DataFrame(with_value, columns=["和值"])
    red1_add_red6 = pd.DataFrame(red1_add_red6, columns=["首尾和值"])
    with_value_data = pd.concat([with_value, red1_add_red6], axis=1)
    return with_value_data



def red6_subtract_red1(red_num):
    """
    功能：red6 - red1
    参数：
        dataset：原始数据集
        red_num_columns：红球的列名
    返回值：red6 - red1 的结果
    """
    red6_subtract_red1_data = red_num["red6"] - red_num["red1"]
    red6_subtract_red1_data = pd.DataFrame(red6_subtract_red1_data, columns=["red6_subtract_red1"])
    return red6_subtract_red1_data



def parity_ratio(red_num):
    """
    功能：计算奇偶比
    参数：
        dataset：原始数据集
        red_num_columns：红球的列名
    返回值：奇数个数、偶数个数、奇偶比
    """
    is_even_number = red_num % 2 == 0  # 是偶数吗，如果是偶数，值为True；如果是奇数，值为False
    odd_even_data = is_even_number.apply(pd.value_counts, axis=1)  # 计算每一行False有几个，True有几个
    odd_even_data.rename(columns={True: "偶数个数", False: "奇数个数"}, inplace=True)  # 修改列名
    odd_even_data = odd_even_data.fillna(0)  # 假如某一行奇数为0，偶数为6，那么这一行的奇数列值为 NAN，所以需要用0填充却失值
    # 计算奇偶比
    odd_even_data.loc[odd_even_data["奇数个数"] == 0, "奇偶比"] = "0比6"
    odd_even_data.loc[odd_even_data["奇数个数"] == 1, "奇偶比"] = "1比5"
    odd_even_data.loc[odd_even_data["奇数个数"] == 2, "奇偶比"] = "2比4"
    odd_even_data.loc[odd_even_data["奇数个数"] == 3, "奇偶比"] = "3比3"
    odd_even_data.loc[odd_even_data["奇数个数"] == 4, "奇偶比"] = "4比2"
    odd_even_data.loc[odd_even_data["奇数个数"] == 5, "奇偶比"] = "5比1"
    odd_even_data.loc[odd_even_data["奇数个数"] == 6, "奇偶比"] = "6比0"
    return odd_even_data



def size_ratio(red_num):
    """
    功能：计算大小比
    参数：
        dataset：原始数据集
        red_num_columns：红球的列名
    返回值：小数个数、大数个数、大小比
    """
    is_small_number = red_num >= 17  # 是大数吗，如果是大数，值为True；如果是小数，值为False
    size_data = is_small_number.apply(pd.value_counts, axis=1)  # 计算每一行False有几个，True有几个
    size_data.rename(columns={True: "大数个数", False: "小数个数"}, inplace=True)  # 修改列名
    size_data = size_data.fillna(0)  # 假如某一行小数为0，大数为6，那么这一行的小数列值为 NAN，所以需要用0填充却失值
    # 计算大小比
    size_data.loc[size_data["大数个数"] == 0, "大小比"] = "0比6"
    size_data.loc[size_data["大数个数"] == 1, "大小比"] = "1比5"
    size_data.loc[size_data["大数个数"] == 2, "大小比"] = "2比4"
    size_data.loc[size_data["大数个数"] == 3, "大小比"] = "3比3"
    size_data.loc[size_data["大数个数"] == 4, "大小比"] = "4比2"
    size_data.loc[size_data["大数个数"] == 5, "大小比"] = "5比1"
    size_data.loc[size_data["大数个数"] == 6, "大小比"] = "6比0"
    return size_data



def qualitative_ratio(red_num):
    """
    功能：计算质合比。质数：[1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    参数：
        dataset：红球数据
    返回值：质数个数、合数个数、质合比
    """
    prime_number_list = []
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        number_of_prime_numbers = 0  # 初始化质数的个数
        # 对 red_num 中的列进行循环
        for column in red_num.columns:
            is_prime_number = red_num.loc[i, column] in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]  # 质数
            if is_prime_number:
                number_of_prime_numbers = number_of_prime_numbers + 1
        prime_number_list.append(number_of_prime_numbers)
    # 质数与合数的个数
    qualitative_ratio_data = pd.DataFrame(prime_number_list, columns=["质数个数"])
    qualitative_ratio_data.loc[:, "合数个数"] = 6 - qualitative_ratio_data.loc[:, "质数个数"]
    # 计算质合比
    qualitative_ratio_data.loc[qualitative_ratio_data["质数个数"] == 0, "质合比"] = "0比6"
    qualitative_ratio_data.loc[qualitative_ratio_data["质数个数"] == 1, "质合比"] = "1比5"
    qualitative_ratio_data.loc[qualitative_ratio_data["质数个数"] == 2, "质合比"] = "2比4"
    qualitative_ratio_data.loc[qualitative_ratio_data["质数个数"] == 3, "质合比"] = "3比3"
    qualitative_ratio_data.loc[qualitative_ratio_data["质数个数"] == 4, "质合比"] = "4比2"
    qualitative_ratio_data.loc[qualitative_ratio_data["质数个数"] == 5, "质合比"] = "5比1"
    qualitative_ratio_data.loc[qualitative_ratio_data["质数个数"] == 6, "质合比"] = "6比0"
    return qualitative_ratio_data



def serial_number(red_num):
    """
    功能：计算是否有连号
    参数：
        dataset：原始数据集
        red_num_columns：红球的列名
    返回值：是否有连号
    """
    serial_number_list = []
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        have_serial_number = []
        # 对 red_num 中的列进行循环
        for j in range(red_num.columns.shape[0]):
            difference = 0
            if j < red_num.columns.shape[0] - 1:
                difference = red_num.loc[i, red_num.columns[j + 1]] - red_num.loc[i, red_num.columns[j]]
            else:
                break
            if difference == 1:
                have_serial_number.append("有连号")
                break
        if len(have_serial_number) == 0:
            have_serial_number.append("无连号")
        serial_number_list.append(have_serial_number)
    serial_number_data = pd.DataFrame(serial_number_list, columns=["连号"])
    return serial_number_data



def repeat_number(red_num):
    """
    功能：计算重号的个数
    参数：
        dataset：原始数据集
        red_num_columns：红球的列名
    返回值：重号的个数
    """
    all_repeat_number = []
    all_repeat_number.append(0)
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        repeat_number_list = []
        if i >= red_num.shape[0] - 1:
            break
        # 对 red_num 中的列进行循环
        for column in red_num.columns:
            is_repeat = red_num.loc[i + 1, column] in red_num.loc[i, :].as_matrix()
            if is_repeat:
                repeat_number_list.append(red_num.loc[i + 1, column])

        all_repeat_number.append(len(repeat_number_list))
    repeat_number_data = pd.DataFrame(all_repeat_number, columns=["重复号码的个数"])
    return repeat_number_data



def except_for_3_remainders(red_num):
    """
    功能：计算除3余数
    参数：
        dataset：原始数据集
        red_num_columns：红球的列名
    返回值：除3余0个数、除3余1个数、除3余2个数
    """
    remainders_number_list = []
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        remainders_0, remainders_1, remainders_2 = 0, 0, 0
        # 对 red_num 中的列进行循环
        for column in red_num.columns:
            if red_num.loc[i, column] in [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]:  # 除3余0
                remainders_0 = remainders_0 + 1
            elif red_num.loc[i, column] in [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31]:  # 除3余1
                remainders_1 = remainders_1 + 1
            elif red_num.loc[i, column] in [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32]:  # 除3余2
                remainders_2 = remainders_2 + 1
        remainders_number_list.append([remainders_0, remainders_1, remainders_2])
    remainders_number_data = pd.DataFrame(remainders_number_list, columns=["除3余0个数", "除3余1个数", "除3余2个数"])
    return remainders_number_data



def road_of_012(red_num):
    """
    功能：计算 012路
    参数：
        dataset：原始数据集
        red_num_columns：红球的列名
    返回值：0路个数、1路个数、2路个数
    """
    # 获取每一列的尾号
    red_num = red_num.applymap(str)  # 把 DataFrame 数据类型转换为 str
    for column in red_num.columns:
        red_num.loc[:, column] = red_num[column].str[-1]
    red_num = red_num.apply(pd.to_numeric)
    # 计算 012路
    road_of_012_list = []
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        road_of_0, road_of_1, road_of_2 = 0, 0, 0
        # 对 red_num 中的列进行循环
        for column in red_num.columns:
            if red_num.loc[i, column] in [0, 3, 6, 9]:  # 0路
                road_of_0 = road_of_0 + 1
            elif red_num.loc[i, column] in [1, 4, 7]:  # 1路
                road_of_1 = road_of_1 + 1
            elif red_num.loc[i, column] in [2, 5, 8]:  # 2路
                road_of_2 = road_of_2 + 1
        road_of_012_list.append([road_of_0, road_of_1, road_of_2])
    road_of_012_data = pd.DataFrame(road_of_012_list, columns=["0路个数", "1路个数", "2路个数"])
    return road_of_012_data



def tail_type(red_num):
    """
    功能：计算尾号的类型
    参数：
        dataset：原始数据集
        red_num_columns：红球的列名
    返回值：尾号类型
    """
    # 获取每一列的尾号
    red_num = red_num.applymap(str)  # 把 DataFrame 数据类型转换为 str
    for column in red_num.columns:
        red_num.loc[:, column] = red_num[column].str[-1]
    # 计算尾号类型
    value_counts_data_1 = red_num.apply(pd.value_counts, axis=1)  # 计算每一行False有几个，True有几个
    value_counts_data_1.rename(columns={
        "0": "尾数为0的个数", "1": "尾数为1的个数", "2": "尾数为2的个数", "3": "尾数为3的个数", "4": "尾数为4的个数",
        "5": "尾数为5的个数", "6": "尾数为6的个数", "7": "尾数为7的个数", "8": "尾数为8的个数", "9": "尾数为9的个数"
    }, inplace=True)  # 修改列名
    value_counts_data_1 = value_counts_data_1.fillna(0)  # 假如某一行小数为0，大数为6，那么这一行的小数列值为 NAN，所以需要用0填充却失值
    # 再一次调用 value_counts() 函数
    value_counts_data_2 = value_counts_data_1.apply(pd.value_counts, axis=1)  # 计算每一行False有几个，True有几个
    del value_counts_data_2[0.0]
    # 计算尾号的类型
    value_counts_data_2.loc[value_counts_data_2[1.0] == 6.0, "尾号类型"] = "ABC"
    value_counts_data_2.loc[value_counts_data_2[2.0] == 1.0, "尾号类型"] = "AA"
    value_counts_data_2.loc[value_counts_data_2[2.0] == 2.0, "尾号类型"] = "AABB"
    value_counts_data_2.loc[value_counts_data_2[3.0] == 1.0, "尾号类型"] = "AAA"
    value_counts_data_2["尾号类型"] = value_counts_data_2["尾号类型"].fillna("其他")
    tail_type_data = pd.DataFrame(value_counts_data_2["尾号类型"], columns=["尾号类型"])
    return tail_type_data



def create_red_feature(red_num, blue_num):
    # 区间比
    interval_data = interval_ratio(red_num)
    # 和值
    with_value_data = with_value(red_num)
    # red6 - red1
    red6_subtract_red1_data = red6_subtract_red1(red_num)
    # 奇偶比
    parity_ratio_data = parity_ratio(red_num)
    # 大小比
    size_data = size_ratio(red_num)
    # 质合比
    qualitative_ratio_data = qualitative_ratio(red_num)
    # 除3余数
    remainders_number_data = except_for_3_remainders(red_num)
    # 012 路
    road_of_012_data = road_of_012(red_num)
    # 连号
    serial_number_data = serial_number(red_num)
    # 重号的个数
    repeat_number_data = repeat_number(red_num)
    # 尾号的类型
    tail_type_data = tail_type(red_num)
    # 把前面大所有特征拼接起来
    red_feature = pd.concat(
        [
            red_num, blue_num, interval_data, with_value_data, red6_subtract_red1_data, parity_ratio_data, size_data,
            qualitative_ratio_data, remainders_number_data, road_of_012_data, serial_number_data, repeat_number_data,
            tail_type_data
        ],
        axis=1
    )
    return red_feature









# 文件路径
data_dir = r"F:\data\SSQ\原始数据"
file_path = os.path.join(data_dir, "开奖表.xlsx")
output_path = os.path.join(data_dir, "feature_red.csv")
# 读取数据
dataset = pd.read_excel(file_path, sheet_name=0)
red_num = dataset[dataset.columns[1:-1]]
blue_num = dataset[dataset.columns[-1]]
# 构造特征
red_feature = create_red_feature(red_num, blue_num)
red_feature.to_csv(output_path, index=False, encoding="gbk")

