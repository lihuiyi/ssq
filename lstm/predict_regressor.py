# -*- coding: utf-8 -*-


import os
import pickle
import pandas as pd
from lstm.lstm_regressor import predict
from lstm.lstm_regressor import plot_results
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
# 模型路径
model_dir = r"F:\SSQ" + "\\" + str(time_step) + "期" + "\\" + label_name + "\\" + label_name + "_model1"



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





# 预测
x_true = x_test
y_true = y_test
global_step_list = [7150]
pkl_dir = data_dir
test_result_dir = r"C:\Users\lenovo\Desktop"

result_csv_path = os.path.join(test_result_dir, model_dir.split("\\")[-1] + "_测试结果.csv")
result_image_path = os.path.join(test_result_dir, model_dir.split("\\")[-1] + "_测试结果.png")
y_pred = predict(model_dir, global_step_list, x_true, y_test=y_true)
y_pred = pd.DataFrame(y_pred, columns=[label_name])
# 标签反转换
pkl_Path = os.path.join(pkl_dir, "数据预处理.pkl")
preprocessing_obj_dict = load_pkl(pkl_Path=pkl_Path)
label_preprocessing_obj = preprocessing_obj_dict["label_preprocessing_obj"]
if label_preprocessing_obj is not None:
    y_pred = label_preprocessing_obj.standardScaler.inverse_transform(y_pred.loc[:, [label_name]])
    y_true = label_preprocessing_obj.standardScaler.inverse_transform(y_true.loc[:, [label_name]])
    y_pred = pd.DataFrame(y_pred, columns=["预测值"])
# 计算误差
y_true = pd.DataFrame(y_true, columns=["真实值"])
data = pd.concat([y_pred, y_true], axis=1)
data = data.round(decimals=0)
data.loc[:, "误差"] = data["真实值"] - data["预测值"]
data = data.round(decimals=0)
# 计算杀号的概率
pc_rate = data.loc[data["误差"] != 0].shape[0] / data.shape[0]
print("\n杀号的概率：", pc_rate, "\n")
# 把预测结果写入到文件
data.to_csv(result_csv_path, index=False, encoding="gbk")
# print(data)
# 画图展示
plot_results(y_true, y_pred, result_image_path)

