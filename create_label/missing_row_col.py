# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd



def init_isometric_matrix():
    """
    功能：初始化等距矩阵
    返回值：等距矩阵的信息（list 类型）
    """
    matrix = np.array(
        [[1, 7, 13, 19, 25, 31],
         [2, 8, 14, 20, 26, 32],
         [3, 9, 15, 21, 27, 33],
         [4, 10, 16, 22, 28, 34],
         [5, 11, 17, 23, 29, 35],
         [6, 12, 18, 24, 30, 36]]
    )
    # 把 matrix 转换为 DataFrame
    column_name = ["A", "B", "C", "D", "E", "F"]
    line_name = ["a", "b", "c", "d", "e", "f"]
    matrix = pd.DataFrame(matrix, columns=column_name)
    isometric_matrix = ["等距矩阵", matrix, column_name, line_name]
    return isometric_matrix



def init_magic_square_matrix_method_1():
    """
    功能：初始化第一种幻方矩阵
    返回值：第一种幻方矩阵的信息（list 类型）
    """
    matrix = np.array(
        [[35, 1, 6, 26, 19, 24],
         [3, 32, 7, 21, 23, 25],
         [31, 9, 2, 22, 27, 20],
         [8, 28, 33, 17, 10, 15],
         [30, 5, 34, 12, 14, 16],
         [4, 36, 29, 13, 18, 11]]
    )
    # 把 matrix 转换为 DataFrame
    column_name = ["A", "B", "C", "D", "E", "F"]
    line_name = ["a", "b", "c", "d", "e", "f"]
    matrix = pd.DataFrame(matrix, columns=column_name)
    magic_square_matrix_method_1 = ["幻方矩阵1", matrix, column_name, line_name]
    return magic_square_matrix_method_1



def init_magic_square_matrix_method_2():
    """
    功能：初始化第二种幻方矩阵
    返回值：第二种幻方矩阵的信息（list 类型）
    """
    matrix = np.array(
        [[4, 13, 27, 36, 29, 2],
         [31, 22, 18, 9, 11, 20],
         [3, 12, 23, 32, 16, 25],
         [30, 21, 14, 5, 7, 34],
         [8, 17, 19, 28, 33, 6],
         [35, 26, 10, 1, 15, 24]]
    )
    # 把 matrix 转换为 DataFrame
    column_name = ["A", "B", "C", "D", "E", "F"]
    line_name = ["a", "b", "c", "d", "e", "f"]
    matrix = pd.DataFrame(matrix, columns=column_name)
    magic_square_matrix_method_2 = ["幻方矩阵2", matrix, column_name, line_name]
    return magic_square_matrix_method_2



def init_ball_matrix():
    """
    功能：初始化装球矩阵
    返回值：装球矩阵的信息（list 类型）
    """
    matrix = np.array(
        [[4, 8, 12, 0, 0, 0, 0, 0, 0, 0],
         [3, 7, 11, 15, 18, 21, 24, 27, 30, 33],
         [2, 6, 10, 14, 17, 20, 23, 26, 29, 32],
         [1, 5, 9, 13, 16, 19, 22, 25, 28, 31]]
    )
    # 把 matrix 转换为 DataFrame
    column_name = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    line_name = ["a", "b", "c", "d"]
    matrix = pd.DataFrame(matrix, columns=column_name)
    ball_matrix = ["装球矩阵", matrix, column_name, line_name]
    return ball_matrix



def init_same_tail_matrix():
    """
    功能：初始化同尾号矩阵
    返回值：同尾号矩阵的信息（list 类型）
    """
    matrix = np.array(
        [[1, 6, 11, 16, 21, 26, 31],
         [2, 7, 12, 7, 22, 27, 32],
         [3, 8, 13, 18, 23, 28, 33],
         [4, 9, 14, 19, 24, 29, 34],
         [5, 10, 15, 20, 25, 30, 35]]
    )
    # 把 matrix 转换为 DataFrame
    column_name = ["A", "B", "C", "D", "E", "F", "G"]
    line_name = ["a", "b", "c", "d", "e"]
    matrix = pd.DataFrame(matrix, columns=column_name)
    same_tail_matrix = ["同尾号矩阵", matrix, column_name, line_name]
    return same_tail_matrix



def init_positive_rotation_matrix():
    """
    功能：初始化正旋矩阵
    返回值：正旋矩阵的信息（list 类型）
    """
    matrix = np.array(
        [[1, 2, 3, 4, 5, 6],
         [20, 21, 22, 23, 24, 7],
         [19, 32, 0, 0, 25, 8],
         [18, 31, 0, 0, 26, 9],
         [17, 30, 29, 28, 27, 10],
         [16, 15, 14, 13, 12, 11]]
    )
    # 把 matrix 转换为 DataFrame
    column_name = ["A", "B", "C", "D", "E", "F"]
    line_name = ["a", "b", "c", "d", "e", "f"]
    matrix = pd.DataFrame(matrix, columns=column_name)
    positive_rotation_matrix = ["正旋矩阵", matrix, column_name, line_name]
    return positive_rotation_matrix



def init_inverse_matrix():
    """
    功能：初始化逆旋矩阵
    返回值：逆旋矩阵（list 类型）
    """
    matrix = np.array(
        [[33, 32, 31, 30, 29, 28],
         [14, 13, 12, 11, 10, 27],
         [15, 2, 0, 0, 9, 26],
         [16, 3, 0, 0, 8, 25],
         [17, 4, 5, 6, 7, 24],
         [18, 19, 20, 21, 22, 23]]
    )
    # 把 matrix 转换为 DataFrame
    column_name = ["A", "B", "C", "D", "E", "F"]
    line_name = ["a", "b", "c", "d", "e", "f"]
    matrix = pd.DataFrame(matrix, columns=column_name)
    inverse_matrix = ["逆旋矩阵", matrix, column_name, line_name]
    return inverse_matrix




def init_all_matrix():
    """
    功能：初始化全部方阵（矩阵）
    返回值：全部方阵（矩阵），是 DataFrame 类型
    """
    isometric_matrix = init_isometric_matrix()  # 初始化等距矩阵
    magic_square_matrix_method_1 = init_magic_square_matrix_method_1()  # 初始化第一种幻方矩阵
    magic_square_matrix_method_2 = init_magic_square_matrix_method_2()  # 初始化第二种幻方矩阵
    ball_matrix = init_ball_matrix()  # 初始化装球矩阵
    same_tail_matrix = init_same_tail_matrix()  # 初始化同尾号矩阵
    positive_rotation_matrix = init_positive_rotation_matrix()  # 初始化正旋矩阵
    inverse_matrix = init_inverse_matrix()  # 初始化逆旋矩阵
    # 得到全部矩阵
    data = [
        isometric_matrix, magic_square_matrix_method_1, magic_square_matrix_method_2, ball_matrix, same_tail_matrix,
        positive_rotation_matrix, inverse_matrix
    ]
    all_matrix = pd.DataFrame(data, columns=["矩阵名称", "矩阵值", "矩阵的列名", "矩阵的行名"])
    return all_matrix



def lack_col(red_num, matrix, column_name):
    """
    功能：获取缺列
    参数：
        red_num：红球
        matrix：方阵（矩阵）
        column_name：方阵（矩阵）的列名
    返回值：缺列的数据
    """
    lack_col_list = []
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        # 初始化缺列字典
        keys = column_name
        values = [0] * len(column_name)
        dic = dict(zip(keys, values))
        # 对 red_num 中的列进行循环
        for number in red_num.loc[i, :]:
            # 对 方阵 中的列进行循环
            for column in matrix.columns.tolist():
                if number in matrix[column].as_matrix():
                    dic[column] = dic[column] + 1
                    break
        lack_col = list(dic.values())  # 得到缺列
        lack_col_list.append(lack_col)  # red_num 中每一行的缺列
    lack_line_data = pd.DataFrame(lack_col_list, columns=column_name)  # 把 red_num 中全部行的缺列转换为 DataFrame
    return lack_line_data



def lack_line(red_num, matrix, line_name):
    """
    功能：获取缺行
    参数：
        red_num：红球
        matrix：方阵（矩阵）
        column_name：方阵（矩阵）的行名
    返回值：缺行的数据
    """
    lack_line_list = []
    # 对 red_num 中的行进行循环
    for i in range(red_num.shape[0]):
        keys = line_name
        values = [0] * len(line_name)
        dic = dict(zip(keys, values))
        # 对 red_num 中的列进行循环
        for number in red_num.loc[i, :]:
            # 对 方阵 中的行进行循环
            for j in range(matrix.shape[0]):
                if number in matrix.loc[j, :].as_matrix():
                    dic[keys[j]] = dic[keys[j]] + 1
                    break
        lack_line = list(dic.values())  # 得到缺行
        lack_line_list.append(lack_line)  # red_num 中每一行的缺行
    lack_line_data = pd.DataFrame(lack_line_list, columns=line_name)  # 把 red_num 中全部行的缺行转换为 DataFrame
    return lack_line_data



def get_missing_row_col(red_num, blue_num, output_dir):
    """
    功能：获取全部方阵（矩阵）的缺列缺行数据，然后写入到csv文件
    参数：
        dataset：包含红球的原始数据
        red_num_columns：6个红球的列名
        output_dir：输出数据的保存目录
    """
    # 创建 output_dir 目录
    is_exists = os.path.exists(output_dir)
    if is_exists is False:
        os.makedirs(output_dir)
    # 初始化全部方阵（矩阵）
    all_matrix = init_all_matrix()
    # 对 all_matrix 进行循环，计算缺列缺行数据
    for i in range(all_matrix.shape[0]):
        matrix = all_matrix.loc[i, "矩阵值"]
        column_name = all_matrix.loc[i, "矩阵的列名"]
        line_name = all_matrix.loc[i, "矩阵的行名"]
        # 计算缺列
        lack_col_data = lack_col(red_num, matrix, column_name)
        # “正旋矩阵”和“逆旋矩阵”在计算缺行时，需要更改第3行第3列的值
        if all_matrix.loc[i, "矩阵名称"] == "正旋矩阵":
            matrix.loc[2, "C"] = 33
        elif all_matrix.loc[i, "矩阵名称"] == "逆旋矩阵":
            matrix.loc[2, "C"] = 1
        # 计算缺行
        lack_line_data = lack_line(red_num, matrix, line_name)
        # 连接 red_num、缺列、缺行
        blue_num = pd.DataFrame(blue_num, columns=["blue"])
        output_data = pd.concat([red_num, blue_num, lack_col_data, lack_line_data], axis=1)
        # 把缺列缺行数据写入到csv文件
        csv_name = all_matrix.loc[i, "矩阵名称"] + ".csv"
        csv_path = os.path.join(output_dir, csv_name)
        output_data.to_csv(csv_path, index=False, encoding="gbk")



# 文件路径
data_dir = r"C:\Users\lenovo\Desktop"
file_path = os.path.join(data_dir, "2011开奖表.xlsx")
# 读取数据
dataset = pd.read_excel(file_path, sheet_name=0)
red_num = dataset[dataset.columns[0:-1]]
blue_num = dataset[dataset.columns[-1]]
# 获取全部方阵（矩阵）的缺列缺行数据
output_dir = r"C:\Users\lenovo\Desktop\output"
get_missing_row_col(red_num, blue_num, output_dir)

