# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split  # 分割训练集与测试集
from utils.preprocessing import Data_preprocessing
from create_time_series.time_series import get_preprocessing_columns
import warnings
warnings.filterwarnings("ignore")



# kill_method
label_name = "A1"
# 时序
time_step = 7
# 文件路径
data_dir = r"F:\data\SSQ" + "\\" + str(time_step) + "期" + "\\" + label_name
file_path = os.path.join(data_dir, "红球杀号对错_时间序列.csv")
train_dir = os.path.join(data_dir, "train_data")
# 分割训练集、验证集、测试集
test_size = 0.15
eval_size = 0.15

# 特征哑编码列
features_onehot_columns = [
    "奇偶比", "大小比", "质合比", "连号", "尾号类型", "蓝球奇偶", "蓝球大小", "蓝球质合",
    "蓝球除3余数", "蓝球除4余数", "蓝球除5余数"
]
features_onehot_columns = get_preprocessing_columns(time_step, features_onehot_columns)
# 特征标准化列
features_standardScaler_columns = [
    'red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue', '三区_1', '三区_2', '三区_3', '三区_断区数',
    '四区_1', '四区_2', '四区_3', '四区_4', '四区_断区数', '和值', '首尾和值', 'red6_subtract_red1',
    '奇数个数', '偶数个数', '小数个数', '大数个数', '质数个数', '合数个数', '除3余0个数', '除3余1个数', '除3余2个数',
    '0路个数', '1路个数', '2路个数', '重复号码的个数', '连号组数', '连号个数', '重号个数', '邻号个数', '孤号个数'
]
features_standardScaler_columns = get_preprocessing_columns(time_step, features_standardScaler_columns)
# 特征归一化列
features_normalizer_columns = None
# 标签预处理方式
label_preprocessing_method = "Onehot"  # 标签预处理方式，"Onehot"、"StandardScaler"



# 读取数据，代码不用改
with open(file_path) as f:
    dataset = pd.read_csv(f)
# 分割训练集、验证集、测试集
dataset = shuffle(dataset, random_state=42)
dataset = dataset.reset_index(drop=True)
stratify = dataset[label_name]
train_data, test_data = train_test_split(dataset, test_size=test_size, shuffle=True, random_state=42, stratify=stratify)
train_data = train_data.reset_index(drop=True)
stratify = train_data[label_name]
train_data, eval_data = train_test_split(train_data, test_size=eval_size, shuffle=True, random_state=42, stratify=stratify)
# 数据预处理
dp = Data_preprocessing()  # 实列化数据预处理对象
train_data = dp.fit_transform(
    train_data, features_onehot_columns, features_standardScaler_columns,
    features_normalizer_columns, label_name, label_preprocessing_method, pkl_dir=train_dir
)
eval_data = dp.transform(eval_data, pkl_dir=train_dir)
test_data = dp.transform(test_data, pkl_dir=train_dir)
# 把预处理后的数据保存到 csv 文件
train_data.to_csv(os.path.join(train_dir, "train.csv"), index=False, encoding="gbk")
eval_data.to_csv(os.path.join(train_dir, "eval.csv"), index=False, encoding="gbk")
test_data.to_csv(os.path.join(train_dir, "test.csv"), index=False, encoding="gbk")

