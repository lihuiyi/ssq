# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd



def create_time_series(dataset, label_name, time_step=7):
    columns = dataset.columns[dataset.columns != label_name].tolist()
    new_columns = []
    for i in range(time_step):
        for column in columns:
            new_columns.append(str(i + 1) + "期_" + column)
    # 拼接七期的数据
    data_list = []
    for i in range(dataset.shape[0]):
        if i >= dataset.shape[0] - time_step:
            break
        time_step_feature = dataset.loc[i: i + time_step - 1, dataset.columns != label_name]
        feature_row = np.reshape(time_step_feature.values, [1, -1])
        row = pd.DataFrame(feature_row, columns=new_columns)
        label = dataset.loc[i + time_step, dataset.columns == label_name]
        row.loc[:, label_name] = label.values
        data_list.append(row)
    time_series = pd.concat(data_list, axis=0)
    return time_series







# kill_method
label_name = "A1"
# 时序
time_step = 3
# 文件路径
data_dir = r"F:\SSQ\train_data_"
feature_path = os.path.join(data_dir, "feature.xlsx")
label_path = os.path.join(data_dir, "红球杀号对错_label.xlsx")
output_dir = r"F:\SSQ" + "\\" + str(time_step) + "期" + "\\" + label_name
output_path = os.path.join(output_dir, "红球杀号对错_时间序列.csv")
# 读取数据
feature = pd.read_excel(feature_path, sheet_name=0)
labels = pd.read_excel(label_path, sheet_name=0)
label = labels[label_name]
dataset = pd.concat([feature, label], axis=1)
# 创建时间序列数据
train_data = create_time_series(dataset, label_name=label_name, time_step=time_step)
train_data.to_csv(output_path, index=False, encoding="gbk")
