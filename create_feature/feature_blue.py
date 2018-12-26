# -*- coding: utf-8 -*-

import os
import pandas as pd



def blue_four_district(blue_num):
    """
    功能：计算蓝球属于哪个区间，分4个区间，01--04为一区、05--08为二区、09--12为三区、13--16为四区
    参数：
        blue_num：蓝球数据
    返回值：蓝球属于哪个区间
    """

    interval_list = []
    # 对 red_num 中的行进行循环
    for i in range(blue_num.shape[0]):
        interval = ""
        if blue_num.loc[i] in  [1, 2, 3, 4]:  # 一区
            interval = "一区"
        elif blue_num.loc[i] in  [5, 6, 7, 8]:  # 二区
            interval = "二区"
        elif blue_num.loc[i] in [9, 10, 11, 12]:  # 三区
            interval = "三区"
        elif blue_num.loc[i] in [13, 14, 15, 16]:  # 四区
            interval = "四区"
        interval_list.append(interval)
    interval_data = pd.DataFrame(interval_list, columns=["蓝球区间"])
    return interval_data



def blue_parity(blue_num):
    """
    功能：计算蓝球属于奇数还是偶数
    参数：
        blue_num：蓝球数据
    返回值：蓝球属于奇数还是偶数
    """
    is_even_number = blue_num % 2 == 0  # 是偶数吗，如果是偶数，值为True；如果是奇数，值为False
    parity_list = []
    # 对 red_num 中的行进行循环
    for i in range(blue_num.shape[0]):
        parity = ""
        if is_even_number.loc[i]:  # 偶数
            parity = "偶数"
        else:   # 奇数
            parity = "奇数"
        parity_list.append(parity)
    parity_data = pd.DataFrame(parity_list, columns=["蓝球奇偶"])
    return parity_data



def blue_size(blue_num):
    """
    功能：计算蓝球属于大数还是小数，01--08为小数,09--16为大数
    参数：
        blue_num：蓝球数据
    返回值：蓝球属于大数还是小数
    """
    size_list = []
    # 对 red_num 中的行进行循环
    for i in range(blue_num.shape[0]):
        size = ""
        if blue_num.loc[i] in  [1, 2, 3, 4, 5, 6, 7, 8]:  # 小数
            size = "小数"
        elif blue_num.loc[i] in [9, 10, 11, 12, 13, 14, 15, 16]:  # 大数
            size = "大数"
        size_list.append(size)
    size_data = pd.DataFrame(size_list, columns=["蓝球大小"])
    return size_data



def blue_qualitative(blue_num):
    """
    功能：计算蓝球属于质数还是合数，质数：[1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    参数：
        blue_num：蓝球数据
    返回值：蓝球属于质数还是合数
    """
    qualitative_list = []
    # 对 blue_num 中的行进行循环
    for i in range(blue_num.shape[0]):
        qualitative = ""
        if blue_num.loc[i] in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:  # 质数
            qualitative = "质数"
        else:  # 合数
            qualitative = "合数"
        qualitative_list.append(qualitative)
    qualitative_data = pd.DataFrame(qualitative_list, columns=["蓝球质合"])
    return qualitative_data




def blue_except_for_3_remainders(blue_num):
    """
    功能：计算蓝球除3余数
    参数：
        blue_num：蓝球数据
    返回值：蓝球除3余数是多少
    """
    except_for_3_remainders = blue_num % 3  # 蓝球除3余数数多少
    remainders_list = []
    # 对 blue_num 中的行进行循环
    for i in range(blue_num.shape[0]):
        remainders = ""
        if except_for_3_remainders.loc[i] == 0:  # 除3余0
            remainders = "余0"
        elif except_for_3_remainders.loc[i] == 1:  # 除3余1
            remainders = "余1"
        elif except_for_3_remainders.loc[i] == 2:  # 除3余2
            remainders = "余2"
        remainders_list.append(remainders)
    remainders_3_data = pd.DataFrame(remainders_list, columns=["蓝球除3余数"])
    return remainders_3_data



def blue_except_for_4_remainders(blue_num):
    """
    功能：计算蓝球除4余数
    参数：
        blue_num：蓝球数据
    返回值：蓝球除4余数是多少
    """
    except_for_4_remainders = blue_num % 4  # 蓝球除3余数数多少
    remainders_list = []
    # 对 blue_num 中的行进行循环
    for i in range(blue_num.shape[0]):
        remainders = ""
        if except_for_4_remainders.loc[i] == 0:  # 除4余0
            remainders = "余0"
        elif except_for_4_remainders.loc[i] == 1:  # 除4余1
            remainders = "余1"
        elif except_for_4_remainders.loc[i] == 2:  # 除4余2
            remainders = "余2"
        elif except_for_4_remainders.loc[i] == 3:  # 除4余3
            remainders = "余3"
        remainders_list.append(remainders)
    remainders_4_data = pd.DataFrame(remainders_list, columns=["蓝球除4余数"])
    return remainders_4_data



def blue_except_for_5_remainders(blue_num):
    """
    功能：计算蓝球除5余数
    参数：
        blue_num：蓝球数据
    返回值：蓝球除5余数是多少
    """
    except_for_5_remainders = blue_num % 5  # 蓝球除3余数数多少
    remainders_list = []
    # 对 blue_num 中的行进行循环
    for i in range(blue_num.shape[0]):
        remainders = ""
        if except_for_5_remainders.loc[i] == 0:  # 除5余0
            remainders = "余0"
        elif except_for_5_remainders.loc[i] == 1:  # 除5余1
            remainders = "余1"
        elif except_for_5_remainders.loc[i] == 2:  # 除5余2
            remainders = "余2"
        elif except_for_5_remainders.loc[i] == 3:  # 除5余3
            remainders = "余3"
        elif except_for_5_remainders.loc[i] == 4:  # 除5余4
            remainders = "余4"
        remainders_list.append(remainders)
    remainders_5_data = pd.DataFrame(remainders_list, columns=["蓝球除5余数"])
    return remainders_5_data



def create_blue_feature(blue_num):
    # 区间
    interval_data = blue_four_district(blue_num)
    # 奇偶
    parit_data = blue_parity(blue_num)
    # 大小
    size_data = blue_size(blue_num)
    # 质合
    qualitative_data = blue_qualitative(blue_num)
    # 除3余数
    remainders_3_data = blue_except_for_3_remainders(blue_num)
    # 除4余数
    remainders_4_data = blue_except_for_4_remainders(blue_num)
    # 除5余数
    remainders_5_data = blue_except_for_5_remainders(blue_num)
    # 把前面大所有特征拼接起来
    blue_feature = pd.concat(
        [interval_data, parit_data, size_data, qualitative_data, remainders_3_data, remainders_4_data, remainders_5_data],
        axis=1
    )
    return blue_feature








# 文件路径
data_dir = r"F:\data\SSQ\原始数据"
file_path = os.path.join(data_dir, "开奖表.xlsx")
output_path = os.path.join(data_dir, "feature_blue.csv")
# 读取数据
dataset = pd.read_excel(file_path, sheet_name=0)
red_num = dataset[dataset.columns[1:-1]]
blue_num = dataset[dataset.columns[-1]]
# 构造特征
blue_feature = create_blue_feature(blue_num)
blue_feature.to_csv(output_path, index=False, encoding="gbk")

