# -*- coding: utf-8 -*-

import os
import pandas as pd
from lstm.lstm_classifier import fit
from lstm.lstm_classifier import predict
from lstm.lstm_regressor import plot_results
import warnings
warnings.filterwarnings("ignore")




# 文件路径
data_dir = r"C:\Users\lenovo\Desktop\新建文件夹\train_data"
train_path = os.path.join(data_dir, "train.csv")
eval_path = os.path.join(data_dir, "eval.csv")
test_path = os.path.join(data_dir, "test.csv")
# 模型超参数
network_structure = "单向多层lstm" # "单向多层lstm"、"双向单层lstm"、"双向多层lstm"
time_step = 28  # 分10个时序
input_size = 28  # 每个时序输入1个特征
hidden_layer_units = 128  # 隐藏层单元数
lstm_layers_num = 2  # LSTM 的层数
n_classes = 2  # 分类数
learning_rate = 0.001
batch_size = 32
weight_decay = 1e-4
keep_prob = 0.5
train_epochs = 300
epochs_per_eval = 100
number_samples = 3200
model_dir = r"C:\Users\lenovo\Desktop\model"



# 读取数据，代码不用改
with open(train_path) as f:
    train_data = pd.read_csv(f)
with open(eval_path) as f:
    eval_data = pd.read_csv(f)
with open(test_path) as f:
    test_data = pd.read_csv(f)
x_train = train_data.loc[:, train_data.columns[0:-n_classes]]
y_train = train_data.loc[:, train_data.columns[-n_classes:]]
x_eval = eval_data.loc[:, eval_data.columns[0:-n_classes]]
y_eval = eval_data.loc[:, eval_data.columns[-n_classes:]]
x_test = test_data.loc[:, test_data.columns[0:-n_classes]]
y_test = test_data.loc[:, test_data.columns[-n_classes:]]



# 训练模型，代码不用改
fit(
    x_train, y_train, x_eval, y_eval, network_structure, time_step, input_size, n_classes, batch_size,
    hidden_layer_units, lstm_layers_num, train_epochs, number_samples,
    weight_decay, learning_rate, keep_prob, epochs_per_eval, model_dir
)



# # 预测
# global_step_list = [380, 390]
# y_pred = predict(model_dir, global_step_list, x_test, y_test)
# # print(y_pred)
# if classifier_or_regressor == "回归":
#     plot_results(y_test , y_pred , r"C:\Users\lenovo\Desktop\测试结果.png")

