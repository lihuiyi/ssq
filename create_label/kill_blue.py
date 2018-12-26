# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd



def kill_rule(kill_data):
    """
    功能：杀号的规则
    参数：
        kill_data：被杀的号码
    """
    # 对 kill_data 中的列进行循环，也就是对每一个杀号方法进行循环
    for column in kill_data.columns:
        kill_data.loc[:, column] = abs(kill_data[column])
        kill_data.loc[kill_data[column] == 0, column] = 10
        # 被杀号码大于33，并且尾数为0时，杀号码10
        # 被杀号码大于33，并且尾数不为0时，杀号码尾数
        condition_1_data = kill_data.loc[kill_data[column] > 16, column]
        condition_2_data = condition_1_data.loc[condition_1_data % 10 == 0]
        condition_3_data = condition_1_data.loc[condition_1_data % 10 != 0]
        condition_2_index = condition_2_data.index
        condition_3_index = condition_3_data.index
        kill_data.loc[condition_2_index, column] = 10  # 被杀号码大于16，并且尾数为0时，杀号码10
        kill_data.loc[condition_3_index, column] = condition_1_data % 10  # 被杀号码大于16，并且尾数不为0时，杀号码尾数
    return kill_data



def method_A(red_num):
    """
    功能：前区号码互减杀号法
    参数：
        red_num：红球
    返回值：被杀的号码
    """
    number_of_methods = 0
    kill_data_list = []
    for i in range(red_num.shape[1]):
        start_columns = "red" + str(i+1)
        start_num = int(start_columns[-1])
        for j in range(start_num, red_num.shape[1]):
            end_columns = "red" + str(j+1)
            result = red_num[end_columns] - red_num[start_columns]  # 计算公式
            result = pd.DataFrame(result.as_matrix(), columns=["A" + str(number_of_methods + 1)])
            kill_data_list.append(result)
            number_of_methods = number_of_methods + 1
    kill_data = pd.concat(kill_data_list, axis=1)
    kill_data = kill_rule(kill_data)
    kill_data = kill_data.drop(kill_data.index[-1], axis=0)
    return kill_data



def method_B(red_num):
    """
    功能：前区号码互加杀号法
    参数：
        red_num：红球
    返回值：被杀的号码
    """
    number_of_methods = 0
    kill_data_list = []
    for i in range(red_num.shape[1]):
        start_columns = "red" + str(i + 1)
        start_num = int(start_columns[-1])
        for j in range(start_num, red_num.shape[1]):
            end_columns = "red" + str(j + 1)
            result = red_num[end_columns] + red_num[start_columns]  # 计算公式
            result = pd.DataFrame(result.as_matrix(), columns=["B" + str(number_of_methods + 1)])
            kill_data_list.append(result)
            number_of_methods = number_of_methods + 1
    kill_data = pd.concat(kill_data_list, axis=1)
    kill_data = kill_rule(kill_data)
    kill_data = kill_data.drop(kill_data.index[-1], axis=0)
    return kill_data



def method_C(red_num):
    """
    功能：前区号码减特定数值杀号法
    参数：
        red_num：红球
    返回值：被杀的号码
    """
    number_of_methods = 0
    kill_data_list = []
    for column in red_num.columns:
        for value in range(1, 60):
            result = red_num[column] - value
            result = pd.DataFrame(result.as_matrix(), columns=["C" + str(number_of_methods + 1)])
            kill_data_list.append(result)
            number_of_methods = number_of_methods + 1
    kill_data = pd.concat(kill_data_list, axis=1)
    kill_data = kill_rule(kill_data)
    kill_data = kill_data.drop(kill_data.index[-1], axis=0)
    return kill_data



def method_D(red_num):
    """
    功能：前区号码加特定数值杀号法
    参数：
        red_num：红球
    返回值：被杀的号码
    """
    number_of_methods = 0
    kill_data_list = []
    for column in red_num.columns:
        for value in range(1, 26):
            result = red_num[column] + value
            result = pd.DataFrame(result.as_matrix(), columns=["D" + str(number_of_methods + 1)])
            kill_data_list.append(result)
            number_of_methods = number_of_methods + 1
    kill_data = pd.concat(kill_data_list, axis=1)
    kill_data = kill_rule(kill_data)
    kill_data = kill_data.drop(kill_data.index[-1], axis=0)
    return kill_data



def method_E(red_num):
    """
    功能：上两期前区对应号码相减
    参数：
        red_num：红球
    返回值：被杀的号码
    """
    kill_data_list = []
    first_row = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    kill_data_list.append(first_row)
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        if i < red_num.shape[0] - 1:
            result = red_num.loc[i + 1] - red_num.loc[i]
            kill_data_list.append(result.as_matrix())
    kill_data = pd.DataFrame(kill_data_list, columns=["E1", "E2", "E3", "E4", "E5", "E6"])
    kill_data = kill_rule(kill_data)
    kill_data = kill_data.drop(kill_data.index[-1], axis=0)
    return kill_data



def method_F(red_num):
    """
    功能：上两期前区对应号码相加
    参数：
        red_num：红球
    返回值：被杀的号码
    """
    kill_data_list = []
    first_row = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    kill_data_list.append(first_row)
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        if i < red_num.shape[0] - 1:
            result = red_num.loc[i + 1] + red_num.loc[i]
            kill_data_list.append(result.as_matrix())
    kill_data = pd.DataFrame(kill_data_list, columns=["F1", "F2", "F3", "F4", "F5", "F6"])
    kill_data = kill_rule(kill_data)
    kill_data = kill_data.drop(kill_data.index[-1], axis=0)
    return kill_data



def method_G(red_num):
    """
    功能：上两期前区对应号码相乘
    参数：
        red_num：红球
    返回值：被杀的号码
    """
    kill_data_list = []
    first_row = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    kill_data_list.append(first_row)
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        if i < red_num.shape[0] - 1:
            result = red_num.loc[i + 1] * red_num.loc[i]
            kill_data_list.append(result.as_matrix())
    kill_data = pd.DataFrame(kill_data_list, columns=["G1", "G2", "G3", "G4", "G5", "G6"])
    kill_data = kill_rule(kill_data)
    kill_data = kill_data.drop(kill_data.index[-1], axis=0)
    return kill_data



def method_H(red_num):
    """
    功能：上两期前区对应号码相除
    参数：
        red_num：红球
    返回值：被杀的号码
    """
    kill_data_list = []
    first_row = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    kill_data_list.append(first_row)
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        if i < red_num.shape[0] - 1:
            result = red_num.loc[i + 1] / red_num.loc[i]
            kill_data_list.append(result.as_matrix())
    kill_data = pd.DataFrame(kill_data_list, columns=["H1", "H2", "H3", "H4", "H5", "H6"])
    # 把 kill_data 四舍五入保留整数，然后把 float 转换为 int
    kill_data = round(kill_data, 0).apply(pd.to_numeric, downcast="integer")
    kill_data = kill_rule(kill_data)
    kill_data = kill_data.drop(kill_data.index[-1], axis=0)
    return kill_data



def method_I(blue_num):
    """
    功能：后区号码减特定数值
    参数：
        red_num：红球
    返回值：被杀的号码
    """
    number_of_methods = 0
    kill_data_list = []
    for value in range(1, 43):
        result = blue_num - value
        result = pd.DataFrame(result.as_matrix(), columns=["I" + str(number_of_methods + 1)])
        kill_data_list.append(result)
        number_of_methods = number_of_methods + 1
    kill_data = pd.concat(kill_data_list, axis=1)
    kill_data = kill_rule(kill_data)
    kill_data = kill_data.drop(kill_data.index[-1], axis=0)
    return kill_data



def method_J(blue_num):
    """
    功能：后区号码加特定数值
    参数：
        red_num：红球
    返回值：被杀的号码
    """
    number_of_methods = 0
    kill_data_list = []
    for value in range(1, 26):
        result = blue_num + value
        result = pd.DataFrame(result.as_matrix(), columns=["J" + str(number_of_methods + 1)])
        kill_data_list.append(result)
        number_of_methods = number_of_methods + 1
    kill_data = pd.concat(kill_data_list, axis=1)
    kill_data = kill_rule(kill_data)
    kill_data = kill_data.drop(kill_data.index[-1], axis=0)
    return kill_data



def method_K(red_num, blue_num):
    """
    功能：前区号码减后区号码
    参数：
        red_num：红球
        blue_num：蓝球
    返回值：被杀的号码
    """
    number_of_methods = 0
    kill_data_list = []
    # 对 red_num 中的列进行循环
    for column in red_num.columns:
        result = red_num[column] - blue_num
        result = pd.DataFrame(result.as_matrix(), columns=["K" + str(number_of_methods + 1)])
        kill_data_list.append(result)
        number_of_methods = number_of_methods + 1
    kill_data = pd.concat(kill_data_list, axis=1)
    kill_data = kill_rule(kill_data)
    kill_data = kill_data.drop(kill_data.index[-1], axis=0)
    return kill_data



def method_L(red_num):
    """
    功能：上两期前区对应号码均值
    参数：
        red_num：红球
    返回值：被杀的号码
    """
    kill_data_list = []
    first_row = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    kill_data_list.append(first_row)
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        if i < red_num.shape[0] - 1:
            result = (red_num.loc[i + 1] + red_num.loc[i]) / 2
            kill_data_list.append(result.as_matrix())
    kill_data = pd.DataFrame(kill_data_list, columns=["L1", "L2", "L3", "L4", "L5", "L6"])
    # 把 kill_data 四舍五入保留整数，然后把 float 转换为 int
    kill_data = round(kill_data, 0).apply(pd.to_numeric, downcast="integer")
    kill_data = kill_rule(kill_data)
    kill_data = kill_data.drop(kill_data.index[-1], axis=0)
    return kill_data



def method_N(red_num, blue_num):
    """
    功能：其他杀号方法
    参数：
        red_num：红球
        blue_num：蓝球
    返回值：被杀的号码
    """
    N1_data = blue_num
    N1_data = pd.DataFrame(N1_data.as_matrix(), columns=["N1"])
    N2_data = blue_num * 2
    N2_data = pd.DataFrame(N2_data.as_matrix(), columns=["N2"])
    N3_data = red_num.mean(axis=1)
    N3_data = pd.DataFrame(N3_data.as_matrix(), columns=["N3"])
    # 把 N3_data 四舍五入保留整数，然后把 float 转换为 int
    N3_data = round(N3_data, 0).apply(pd.to_numeric, downcast="integer")
    N4_data = N3_data["N3"] - blue_num
    N4_data = pd.DataFrame(N4_data.as_matrix(), columns=["N4"])
    # 计算 N5_data、N6_data
    N5_data_list, N6_data_list = [], []
    first_row = np.nan
    N5_data_list.append(first_row)
    N6_data_list.append(first_row)
    for i in range(red_num.shape[0]):
        if i < red_num.shape[0] - 1:
            red6_column_name = red_num.columns[-1]
            red1_column_name = red_num.columns[0]
            N5_result = red_num.loc[i + 1, red6_column_name] - red_num.loc[i, red1_column_name]
            N6_result = red_num.loc[i + 1, red1_column_name] - red_num.loc[i, red6_column_name]
            N5_data_list.append(N5_result)
            N6_data_list.append(N6_result)
    N5_data = pd.DataFrame(N5_data_list, columns=["N5"])
    N6_data = pd.DataFrame(N6_data_list, columns=["N6"])
    # 计算 N7_data、N8_data、N9_data
    N7_data_list, N8_data_list, N9_data_list = [], [], []
    first_row = np.nan
    N7_data_list.append(first_row)
    N8_data_list.append(first_row)
    N9_data_list.append(first_row)
    for i in range(blue_num.shape[0]):
        if i < blue_num.shape[0] - 1:
            N7_result = blue_num.loc[i + 1] - blue_num.loc[i]
            N8_result = blue_num.loc[i + 1] + blue_num.loc[i]
            N9_result = (blue_num.loc[i + 1] + blue_num.loc[i]) / 2
            N7_data_list.append(N7_result)
            N8_data_list.append(N8_result)
            N9_data_list.append(N9_result)
    N7_data = pd.DataFrame(N7_data_list, columns=["N7"])
    N8_data = pd.DataFrame(N8_data_list, columns=["N8"])
    N9_data = pd.DataFrame(N9_data_list, columns=["N9"])
    # 把 N9_data 四舍五入保留整数，然后把 float 转换为 int
    N9_data = round(N9_data, 0).apply(pd.to_numeric, downcast="integer")
    # method_N()函数中全部的杀号方法
    kill_data = pd.concat([N1_data, N2_data, N3_data, N4_data, N5_data, N6_data, N7_data, N8_data, N9_data], axis=1)
    kill_data = kill_rule(kill_data)
    kill_data = kill_data.drop(kill_data.index[-1], axis=0)
    return kill_data



def kill_number(red_num, blue_num):
    """
    功能：全部杀号方法的汇集
    参数：
        red_num：红球
    返回值：被杀的号码，杀号对错、杀号准确率
    """
    method_A_data = method_A(red_num)
    method_B_data = method_B(red_num)
    method_C_data = method_C(red_num)
    method_D_data = method_D(red_num)
    method_E_data = method_E(red_num)
    method_F_data = method_F(red_num)
    method_G_data = method_G(red_num)
    method_H_data = method_H(red_num)
    method_I_data = method_I(blue_num)
    method_J_data = method_J(blue_num)
    method_K_data = method_K(red_num, blue_num)
    method_L_data = method_L(red_num)
    method_N_data = method_N(red_num, blue_num)
    data_list = [
        method_A_data, method_B_data, method_C_data, method_D_data, method_E_data, method_F_data, method_G_data,
        method_H_data, method_I_data, method_J_data, method_K_data, method_L_data, method_N_data
    ]
    kill_data = pd.concat(data_list, axis=1)
    kill_data = round(kill_data, 0).apply(pd.to_numeric, downcast="integer")

    # 向 kill_data 的第一行插入 NAN
    first_row = np.array([[np.nan] * kill_data.shape[1]])
    first_row = pd.DataFrame(first_row, columns=kill_data.columns.tolist())
    kill_data = pd.concat([first_row, kill_data], axis=0)
    kill_data = kill_data.reset_index(drop=True)

    return kill_data



def judge_right_or_not(blue_num, kill_data):
    """
    功能：判断每一种杀号方法是对还是错
    参数：
        red_num：红球
        kill_data：每一中杀号方法的被杀号码杀号
    返回值：每一种杀号方法是对还是错
    """
    correct_or_wrong_list = []
    # 对 kill_data 中的列进行循环，也就是对每一个杀号方法进行循环
    for column in kill_data.columns:
        print(column)
        column_data = kill_data[column]
        correct_or_wrong = []
        # 对 kill_data 中的行进行循环
        for i in range(kill_data.shape[0]):
            # 计算杀号方法有没有杀对号码
            is_not_in = (column_data.loc[i] != blue_num.loc[i])
            is_nan = False
            if i == 0 or i == 1:
                is_nan = pd.Series([column_data.loc[i]]).isnull().values[0]
            if is_nan:
                correct_or_wrong.append(np.nan)
            elif is_not_in:
                correct_or_wrong.append("对")
            else:
                correct_or_wrong.append("错")
        correct_or_wrong_df = pd.DataFrame(correct_or_wrong, columns=[column])
        correct_or_wrong_list.append(correct_or_wrong_df)
    correct_or_wrong_data = pd.concat(correct_or_wrong_list, axis=1)
    return correct_or_wrong_data



def get_accuracy(correct_or_wrong_data):
    """
    功能：获取全部杀号方法的准确率
    参数：
        correct_or_wrong_data：每一种杀号方法是对还是错
    返回值：全部杀号方法的准确率
    """
    # 计算每一列 对的有几个，错的有几个
    value_counts_data = correct_or_wrong_data.apply(pd.value_counts)
    # 计算每一列的准确率
    accuracy = value_counts_data.loc["对"] / (value_counts_data.loc["对"] + value_counts_data.loc["错"])
    accuracy = round(accuracy, 2)  # 四舍五入保留2位小数
    accuracy_data = pd.DataFrame({"杀号方法": accuracy.keys().tolist(), "准确率": accuracy.values})
    return accuracy_data



def to_execl(correct_or_wrong_data, accuracy_data, output_dir):
    """
    功能：把杀号结果写入 execl
    参数：
        correct_or_wrong_data：每一种杀号方法是对还是错
        accuracy_data：全部杀号方法的准确率
    """
    is_exists = os.path.exists(output_dir)  # 判断一个目录是否存在
    if is_exists is False:
        os.makedirs(output_dir)  # 创建目录
    output_path1 = os.path.join(output_dir, "杀号对错.xlsx")
    output_path2 = os.path.join(output_dir, "杀号准确率.xlsx")
    correct_or_wrong_data.to_excel(output_path1, index=False, encoding="gbk")
    accuracy_data.to_excel(output_path2, index=False, encoding="gbk")






# 文件路径
data_dir = r"C:\Users\lenovo\Desktop"
output_dir = r"C:\Users\lenovo\Desktop\blue_output"
file_path = os.path.join(data_dir, "2011开奖表.xlsx")
# 读取数据
dataset = pd.read_excel(file_path, sheet_name=0)
red_num = dataset[dataset.columns[0:-1]]
blue_num = dataset[dataset.columns[-1]]
# 杀号
kill_data = kill_number(red_num, blue_num)
kill_data.to_csv(r"C:\Users\lenovo\Desktop\blue_kill_data.csv", index=False, encoding="gbk")
# 判断每一种杀号方法是对还是错
correct_or_wrong_data = judge_right_or_not(blue_num, kill_data)
# 计算每一种杀号方法的准确率
accuracy_data = get_accuracy(correct_or_wrong_data)
# 把杀号结果写入到 excel
to_execl(correct_or_wrong_data, accuracy_data, output_dir)

