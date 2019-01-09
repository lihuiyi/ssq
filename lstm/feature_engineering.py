# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split  # 分割训练集与测试集
from utils.preprocessing import Data_preprocessing
import warnings
warnings.filterwarnings("ignore")



def create_time_series(dataset, label_name, time_step=7):
    columns = dataset.columns[dataset.columns != label_name].tolist()
    new_columns = []
    for i in range(time_step):
        for column in columns:
            new_columns.append(str(i + 1) + "期_" + column)
    # 拼接七期的数据
    data_list = []
    for j in range(dataset.shape[0]):
        if j >= dataset.shape[0] - time_step:
            break
        # 构造特征
        time_step_feature = dataset.loc[j: j + time_step - 1, dataset.columns != label_name]
        row_feature = np.reshape(time_step_feature.values, [1, -1])
        row_feature = pd.DataFrame(row_feature, columns=new_columns)
        # 构造标签
        row_label = dataset.loc[j + time_step, dataset.columns == label_name]
        row_label = pd.DataFrame(row_label.values, columns=[label_name])
        # 拼接特征和标签
        row = pd.concat([row_feature, row_label], axis=1)
        # 把每一行数据添加到 data_list 中
        data_list.append(row)
    # 把全部行的数据拼接起来
    time_series = pd.concat(data_list, axis=0)
    return time_series





# 时序
time_step = 14
# kill_method
label_name = "A1"
# 文件路径
data_dir = r"F:\SSQ" + "\\" + str(time_step) + "期" + "\\" + label_name
file_path = os.path.join(data_dir, "data.xlsx")
train_dir = os.path.join(data_dir, "train_data")
# 分割训练集、验证集、测试集
test_size = 0.15
eval_size = 0.15
# 特征哑编码列
features_onehot_columns = [
    "奇偶比", "大小比", "质合比", "尾号类型", "蓝球奇偶", "蓝球大小", "蓝球质合",
    "蓝球除3余数", "蓝球除4余数", "蓝球除5余数"
]
# 特征标准化列
features_standardScaler_columns = [
    'red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue', '三区_1', '三区_2', '三区_3', '三区_断区数',
    '四区_1', '四区_2', '四区_3', '四区_4', '四区_断区数', '和值', '首尾和值', 'red6_subtract_red1',
    '奇数个数', '偶数个数', '小数个数', '大数个数', '质数个数', '合数个数', '除3余0个数', '除3余1个数', '除3余2个数',
    '0路个数', '1路个数', '2路个数', '连号组数', '连号个数', '重号个数', '邻号个数', '孤号个数'
]
# 特征归一化列
features_normalizer_columns = None
# 标签预处理方式
label_preprocessing_method = "StandardScaler"  # 标签预处理方式，None、"Onehot"、"StandardScaler"



# 下面代码不用改
# 读取数据
dataset = pd.read_excel(file_path, sheet_name=0)
# 分割训练集、验证集、测试集
# dataset = shuffle(dataset, random_state=42)
# dataset = dataset.reset_index(drop=True)
train_data, test_data = train_test_split(dataset, test_size=test_size, shuffle=False, random_state=None)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
train_data, eval_data = train_test_split(train_data, test_size=eval_size, shuffle=False, random_state=None)
train_data = train_data.reset_index(drop=True)
eval_data = eval_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
# 数据预处理
dp = Data_preprocessing()  # 实列化数据预处理对象
train_data = dp.fit_transform(
    train_data, features_onehot_columns, features_standardScaler_columns,
    features_normalizer_columns, label_name, label_preprocessing_method, pkl_dir=train_dir
)
eval_data = dp.transform(eval_data, pkl_dir=train_dir)
test_data = dp.transform(test_data, pkl_dir=train_dir)
# 创建时间序列
train_data = create_time_series(train_data, label_name=label_name, time_step=time_step)
eval_data = create_time_series(eval_data, label_name=label_name, time_step=time_step)
test_data = create_time_series(test_data, label_name=label_name, time_step=time_step)
# 把预处理后的数据保存到 csv 文件
train_data.to_csv(os.path.join(train_dir, "train.csv"), index=False, encoding="gbk")
eval_data.to_csv(os.path.join(train_dir, "eval.csv"), index=False, encoding="gbk")
test_data.to_csv(os.path.join(train_dir, "test.csv"), index=False, encoding="gbk")

