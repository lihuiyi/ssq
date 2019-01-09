# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
from lstm.lstm_classifier import fit
from lstm.lstm_classifier import predict
import warnings
warnings.filterwarnings("ignore")




time_step = 3
label_name = "A1"
# 文件路径
data_dir = r"F:\SSQ" + "\\" + str(time_step) + "期" + "\\" + label_name + "\\" + "train_data"
train_path = os.path.join(data_dir, "train.csv")
eval_path = os.path.join(data_dir, "eval.csv")
test_path = os.path.join(data_dir, "test.csv")
# 模型超参数
network_structure = "双向单层lstm" # "单向多层lstm"、"双向单层lstm"、"双向多层lstm"
hidden_layer_units = 128  # 隐藏层单元数
lstm_layers_num = 2  # LSTM 的层数
n_classes = 2  # 分类数
learning_rate = 0.001
batch_size = 32
weight_decay = 3e-3
keep_prob = 0.5
train_epochs = 100
epochs_per_eval = 5
model_dir = r"F:\SSQ" + "\\" + str(time_step) + "期" + "\\" + label_name + "\\" + "model"



# 读取数据，代码不用改
with open(train_path) as f:
    train_data = pd.read_csv(f)
with open(eval_path) as f:
    eval_data = pd.read_csv(f)
x_train = train_data.loc[:, train_data.columns[0:-n_classes]]
y_train = train_data.loc[:, train_data.columns[-n_classes:]]
x_eval = eval_data.loc[:, eval_data.columns[0:-n_classes]]
y_eval = eval_data.loc[:, eval_data.columns[-n_classes:]]




# 训练模型，代码不用改
number_samples = x_train.shape[0]
input_size = (x_train.shape[1] - n_classes) / time_step
input_size = int(input_size)
fit(
    x_train, y_train, x_eval, y_eval, network_structure, time_step, input_size, n_classes, batch_size,
    hidden_layer_units, lstm_layers_num, train_epochs, number_samples,
    weight_decay, learning_rate, keep_prob, epochs_per_eval, model_dir
)




# # 预测
# with open(test_path) as f:
#     test_data = pd.read_csv(f)
# x_test = test_data.loc[:, test_data.columns[0:-n_classes]]
# y_test = test_data.loc[:, test_data.columns[-n_classes:]]
#
#
# global_step_list_list = [[975], [1425], [750], [1050], [75], [150], [375], [600], [150], [900], [1425]]
# y_pred_list = []
# for i in range(11):
#     current_model_dir = model_dir + str(i + 1)
#     scope_name = "LSTM" + str(i + 1)
#     global_step_list = global_step_list_list[i]
#     y_pred = predict(current_model_dir, global_step_list, x_test, y_test, scope_name)
#     y_pred_list.append(y_pred)
# if len(y_pred_list) > 1:
#     y_pred_multi_model = pd.concat(y_pred_list, axis=1)  # 多个模型的预测值矩阵，每一列就是一个模型的预测值
#     y_pred_value_counts = y_pred_multi_model.apply(pd.value_counts, axis=1)  # 计算每一行唯一值有几个
#     y_pred = y_pred_value_counts.fillna(0).idxmax(axis=1)  # 用0填充却失值，然后计算每一行中出现次数最多的值，作为最终的预测值
# else:
#     y_pred = y_pred_list[0]
# test_auc = roc_auc_score(y_true=np.argmax(y_test.values, 1), y_score=y_pred.values)
# test_recall = recall_score(y_true=np.argmax(y_test.values, 1), y_pred=y_pred.values)
# test_matrix = confusion_matrix(y_true=np.argmax(y_test.values, 1), y_pred=y_pred.values)
# test_wsl = test_matrix[0, 1] / (test_matrix[0, 0] + test_matrix[0, 1])
# print("AUC：" + str(test_auc), "  召回率" + str(test_recall), "   误杀率：" + str(test_wsl))
# print(test_matrix)
