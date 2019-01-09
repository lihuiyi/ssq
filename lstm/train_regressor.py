# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd
from lstm.lstm_regressor import fit
import warnings
warnings.filterwarnings("ignore")



def load_pkl(pkl_Path):
    with open(pkl_Path , 'rb') as f:
        pkl_obj = pickle.load(f)
    return pkl_obj





time_step = 14
label_name = "A1"
# 文件路径
data_dir = r"F:\SSQ" + "\\" + str(time_step) + "期" + "\\" + label_name + "\\" + "train_data"
train_path = os.path.join(data_dir, "train.csv")
eval_path = os.path.join(data_dir, "eval.csv")
test_path = os.path.join(data_dir, "test.csv")
# 模型超参数
network_structure = "单向多层lstm" # "单向多层lstm"、"双向单层lstm"、"双向多层lstm"
# 128*2 100*2 156*2 128*1
hidden_layer_units = 128  # 隐藏层单元数
lstm_layers_num = 5  # LSTM 的层数
learning_rate = 0.001
batch_size = 64
weight_decay = 1e-40
keep_prob = 0.9
train_epochs = 300
epochs_per_eval = 5
model_dir = r"F:\SSQ" + "\\" + str(time_step) + "期" + "\\" + label_name + "\\" + label_name + "_model2"



# 读取数据，代码不用改
with open(train_path) as f:
    train_data = pd.read_csv(f)
with open(eval_path) as f:
    eval_data = pd.read_csv(f)
with open(test_path) as f:
    test_data = pd.read_csv(f)
x_train = train_data.loc[:, train_data.columns[0:-1]]
y_train = train_data.loc[:, train_data.columns[-1:]]
x_eval = eval_data.loc[:, eval_data.columns[0:-1]]
y_eval = eval_data.loc[:, eval_data.columns[-1:]]
x_test = test_data.loc[:, test_data.columns[0:-1]]
y_test = test_data.loc[:, test_data.columns[-1:]]



# 训练模型，代码不用改
number_samples = train_data.shape[0]
input_size = x_train.shape[1] / time_step
input_size = int(input_size)
fit(
    x_train, y_train, x_eval, y_eval, network_structure, time_step, input_size, batch_size,
    hidden_layer_units, lstm_layers_num, train_epochs, number_samples,
    weight_decay, learning_rate, keep_prob, epochs_per_eval, model_dir
)


