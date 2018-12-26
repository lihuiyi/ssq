# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



hyper_parameter_pkl_name = "神经网络超参数.pkl"
network_structure_ = "network_structure"
time_step_ = "time_step"
input_size_ = "input_size"
hidden_layer_units_ = "hidden_layer_units"
lstm_layers_num_ = "lstm_layers_num"
n_classes_ = "n_classes"
keep_prob_ = "keep_prob"
learning_rate_ = "learning_rate"
weight_decay_ = "weight_decay"
batch_size_ = "batch_size"
train_epochs_ = "train_epochs"
epochs_per_eval_ = "epochs_per_eval"



def one_way_multilayer_lstm(lstm_input, hidden_layer_units, lstm_layers_num, keep_prob_placeholder):
    """
    功能：搭建 单向多层 LSTM 网络结构
    参数：
        lstm_input：lstm 网络的输入
        hidden_layer_units：隐藏层单元数
        lstm_layers_num：隐藏层数
        keep_prob_placeholder：Dropout 的保留率，是 placeholder 类型
    返回值：全连接层的输入
    """
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob_placeholder)  # dropout 操作
    multi_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(lstm_layers_num)], state_is_tuple=True)  # 多层 LSTM
    # init_state = multi_lstm.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(multi_lstm, lstm_input, dtype=tf.float32)  # LSTM 前向传播
    # 得到全连接层的输入
    outputs = tf.transpose(outputs, (1, 0, 2))
    fc_input = outputs[-1]  # 把最后一个状态的输出作为全连接层的输入
    return fc_input



def two_way_single_layer_lstm(lstm_input, hidden_layer_units, keep_prob_placeholder):
    """
    功能：搭建 双向单层 LSTM 网络结构
    参数：
        lstm_input：lstm 网络的输入
        hidden_layer_units：隐藏层单元数
        lstm_layers_num：隐藏层数
        keep_prob_placeholder：Dropout 的保留率，是 placeholder 类型
    返回值：全连接层的输入
    """
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_units, forget_bias=1.0, state_is_tuple=True)
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob_placeholder)  # dropout 操作
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_units, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob_placeholder)  # dropout 操作
    outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, lstm_input, dtype=tf.float32)
    outputs = tf.concat(outputs, 2)  # 将前向和后向的状态连接起来
    # 得到全连接层的输入
    outputs = tf.transpose(outputs, (1, 0, 2))
    fc_input = outputs[-1]  # 把最后一个状态的输出作为全连接层的输入
    return fc_input



def two_way_multilayer_lstm(lstm_input, hidden_layer_units, lstm_layers_num, keep_prob_placeholder):
    """
    功能：搭建 双向多层 LSTM 网络结构
    参数：
        lstm_input：lstm 网络的输入
        hidden_layer_units：隐藏层单元数
        lstm_layers_num：隐藏层数
        keep_prob_placeholder：Dropout 的保留率，是 placeholder 类型
    返回值：全连接层的输入
    """
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_units, forget_bias=1.0, state_is_tuple=True)
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob_placeholder)  # dropout 操作
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_units, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob_placeholder)  # dropout 操作
    # 建立前向和后向的三层RNN
    Gmcell = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell for _ in range(lstm_layers_num)])
    Gmcell_bw = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell for _ in range(lstm_layers_num)])
    outputs, sGoutput_state_fw, sGoutput_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        [Gmcell], [Gmcell_bw], lstm_input, dtype=tf.float32
    )
    outputs = tf.concat(outputs, 2)  # 将前向和后向的状态连接起来
    # 得到全连接层的输入
    outputs = tf.transpose(outputs, (1, 0, 2))
    fc_input = outputs[-1]  # 把最后一个状态的输出作为全连接层的输入
    return fc_input



def forward_propagation(network_structure, features, time_step, input_size,
                        hidden_layer_units, lstm_layers_num, n_classes, keep_prob_placeholder):
    """
    功能：LSTM 前向传播
    参数：
        features：特征，shape = [batch_size, time_step, input_size]
        time_step：时间阶段
        input_size：每一个阶段的输入维度
        hidden_layer_units：隐藏层单元数
        lstm_layers_num：隐藏层数
        n_classes：分类数
        batch_size：就是 batch_size
        keep_prob_placeholder：Dropout 的保留率，是 placeholder 类型
    返回值：前向传播的输出值
    """
    # 把 x 变换成 LSTM 需要的格式
    features = tf.reshape(features, [-1, time_step, input_size])  # shape = [batch_size, time_step, input_size]
    features = tf.reshape(features, [-1, input_size])  # shape = [batch_size * time_step, input_size]

    # 输入层 到 隐藏层进行全连接，把全部时序一起运算
    hidden_layer_output = tf.layers.dense(
        inputs=features,
        units=hidden_layer_units,
        activation=tf.nn.relu,
        kernel_initializer=tf.variance_scaling_initializer()
    )
    # 把隐藏层的输出切分成 time_step 个时序，然后作为 LSTM 的输入
    lstm_input = tf.reshape(hidden_layer_output , [-1 , time_step , hidden_layer_units])

    # LSTM 层
    if network_structure == "单向多层lstm":
        fc_input = one_way_multilayer_lstm(lstm_input, hidden_layer_units, lstm_layers_num, keep_prob_placeholder)
    elif network_structure == "双向单层lstm":
        fc_input = two_way_single_layer_lstm(lstm_input, hidden_layer_units, keep_prob_placeholder)
    elif network_structure == "双向多层lstm":
        fc_input = two_way_multilayer_lstm(lstm_input, hidden_layer_units, lstm_layers_num, keep_prob_placeholder)
    else:
        fc_input = None

    # LSTM 与输出层进行全连接
    logits = tf.layers.dense(
        inputs=fc_input,
        units=n_classes,
        kernel_initializer=tf.variance_scaling_initializer()
    )
    return logits



def backward_propagation(labels, logits, weight_decay, learning_rate, global_step):
    """
    功能：LSTM 反向传播
    参数：
        labels：标签，需要哑编码
        logits：前向传播的输出值
        weight_decay：权重衰减系数，也就是 L2 正则化系数
        learning_rate：学习率
        global_step：就是 global_step
    返回值：train_op
    """
    # 定义损失函数
    hubers = tf.losses.huber_loss(labels=labels, predictions=logits)
    error = tf.reduce_sum(hubers)
    tf.summary.scalar("error", error)
    l2_loss = weight_decay * tf.add_n(
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()]
    )
    loss = error + l2_loss
    tf.summary.scalar("loss", loss)
    # 定义优化器，然后最小化损失函数
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, error



def next_batch(current_iteration_num, number_samples, features, labels, batch_size):
    """
    功能：获取当前迭代时的 batch 数据
    参数：
        current_iteration_num：当前是第几次迭代
        number_samples：训练样本的数量
        features：特征
        labels：标签
        batch_size：就是 batch_size
    返回值：当前迭代时的 batch 数据
    """
    # 每个 epochs 需要的迭代次数
    one_epochs_steps = (number_samples / batch_size)
    one_epochs_steps = int(round(one_epochs_steps, 0))
    # 一个 epochs 中的第几个 batch 数据
    n_batch = current_iteration_num % one_epochs_steps
    # 计算 start_index、 end_index
    start_index = (n_batch * batch_size) % number_samples #当前batch的起始index
    end_index = min(start_index + batch_size  , number_samples) #当前batch的结束index
    # 每个 epochs 开始时都洗牌
    train_data = pd.concat([features, labels], axis=1)
    train_data = shuffle(train_data)
    train_data = train_data.reset_index(drop=True)
    if len(labels.shape) == 2:
        n_classes = labels.shape[1]
    else:
        n_classes = 1
    features = train_data.loc[:, train_data.columns[0:-n_classes]]
    labels = train_data.loc[:, train_data.columns[-n_classes:]]
    # 获取 batch_x、 batch_y
    batch_x = features[start_index: end_index] #当前batch的起始数据
    batch_y = labels[start_index: end_index] #当前batch的结束数据
    return batch_x, batch_y



def train(train_op, error, features, labels, logits, x_train, y_train, x_eval, y_eval, train_epochs, number_samples,
          batch_size, keep_prob_placeholder, keep_prob, epochs_per_eval, model_dir, global_step):
    """
    功能：训练
    参数：
        train_op：就是 train_op
        features：特征，placeholder 类型
        labels：标签，placeholder 类型
        logits：LSTM 的输出值
        x_train：训练集的特征，需要在 sess.run() 时，把特征传进去
        y_train：训练集的标签，需要在 sess.run() 时，把标签传进去
        x_eval：验证集的特征，需要在 sess.run() 时，把特征传进去
        y_eval：验证集的标签，需要在 sess.run() 时，把标签传进去
        train_epochs：训练多少个 epoch
        number_samples：训练集样本数量
        batch_size：就是 batch_size
        keep_prob_placeholder：Dropout 的保留率，placeholder 类型
        keep_prob：Dropout 的保留率，需要在 sess.run() 时，把 keep_prob 传进去
        epochs_per_eval：验证的周期，每迭代多少次使用验证集验证一下，并且保存模型
        model_dir：模型保存的目录
        global_step：就是 global_step
    返回值：没有返回值
    """
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=None)
        # 初始化所有变量
        sess.run(tf.global_variables_initializer())
        # tensorboard
        writer_train_summary, writer_eval_summary = writer_summary(sess, model_dir)
        tensorboard_summary = tf.summary.merge_all()  # 把代码中全部 tf.summary("某命名空间" , value) 加到 tensorboard_summary 中
        # 每个 epochs 需要的迭代次数
        one_epochs_steps = (number_samples / batch_size)
        one_epochs_steps = int(round(one_epochs_steps, 0))
        # 迭代训练神经网络
        max_steps = train_epochs * one_epochs_steps
        for i in range(max_steps):
            # 训练
            batch_x, batch_y = next_batch(i, number_samples, x_train, y_train, batch_size)
            sess.run(train_op, feed_dict={features:batch_x , labels:batch_y , keep_prob_placeholder:keep_prob})
            # 验证训练集
            steps = epochs_per_eval * one_epochs_steps
            if (i + 1) % steps == 0:
                train_error, train_summary = sess.run(
                    [error, tensorboard_summary],
                    feed_dict={features: batch_x, labels: batch_y, keep_prob_placeholder: 1}
                )
                writer_train_summary.add_summary(train_summary, i + 1)  # 每 i+1 次 把 summ 加到 tensorboard 中
                # 验证验证集
                eval_error, eval_summary = sess.run(
                    [error, tensorboard_summary],
                    feed_dict={features: x_eval, labels: y_eval, keep_prob_placeholder: 1}
                )
                writer_eval_summary.add_summary(eval_summary, i + 1)  # 每 i+1 次 把 summ 加到 tensorboard 中
                print(str(i + 1) + "   训练集：" + str(train_error) + "   验证集" + str(eval_error))
                # 保存模型
                save_module(saver, sess, model_dir, global_step)



def save_module(saver , sess , modeldir, global_step, model_name="model.ckpt"):
    """
    功能：保存模型
    参数：
        saver：tf.train.Saver(max_to_keep=None)
        sess：tf.Session() as sess
        modeldir：模型保存的目录
        global_step：就是 global_step
    返回值：没有返回值
    """
    is_exists = os.path.exists(modeldir)  # 判断一个目录是否存在
    if is_exists is False:
        os.makedirs(modeldir)  # 创建目录
    model_path = os.path.join(modeldir, model_name)
    saver.save(sess, model_path , global_step=global_step)



def writer_summary(sess, modeldir):
    """
    功能：定义 tensorboard 中的 writer 对象
    参数：
        sess：tf.Session() as sess
        modeldir：模型保存的目录
    返回值：
        训练集的 writer 对象
        验证集的 writer 对象
    """
    is_exists = os.path.exists(modeldir + r"\eval")  # 判断一个目录是否存在
    if is_exists is False:
        os.makedirs(modeldir + r"\eval")  # 创建目录
    writer_train_summary = tf.summary.FileWriter(modeldir , sess.graph)  # 写入 tensorboard 的核心代码
    writer_eval_summary = tf.summary.FileWriter(modeldir + r"\eval" , sess.graph)  # 写入 tensorboard 的核心代码
    #把 write_data 中的内容写入 .bat 文件
    bat_filePath = modeldir + r"\启动tensorboard.bat"
    write_data = "@echo off\ncd ..\nset pard=%cd%\npopd\ncd %pard%\ntensorboard --logdir=" + modeldir.split("\\")[-1]
    with open(bat_filePath , "w") as file:  # 第一个参数是文件路径。第二个参数是"w"，代表写入。最后赋值给file对象
        file.write(write_data)  # file对象调用write()函数，把变量a中的字符串写入文件
    return writer_train_summary, writer_eval_summary



def save_hyper_parameter(model_dir, network_structure, time_step, input_size, hidden_layer_units, lstm_layers_num, n_classes, keep_prob,
                         learning_rate, weight_decay, batch_size, train_epochs, epochs_per_eval):
    parameter_pkl_obj = {
        network_structure_ : network_structure,
        time_step_ : time_step,
        input_size_ : input_size,
        hidden_layer_units_ : hidden_layer_units,
        lstm_layers_num_ : lstm_layers_num,
        n_classes_ : n_classes,
        keep_prob_ : keep_prob,
        learning_rate_ : learning_rate,
        weight_decay_ : weight_decay,
        batch_size_ : batch_size,
        train_epochs_ : train_epochs,
        epochs_per_eval_ : epochs_per_eval
    }
    pkl_path = os.path.join(model_dir, hyper_parameter_pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(parameter_pkl_obj, f)



def fit(x_train, y_train, x_eval, y_eval, network_structure, time_step, input_size, n_classes, batch_size,
        hidden_layer_units, lstm_layers_num, train_epochs, number_samples,
        weight_decay, learning_rate, keep_prob, epochs_per_eval, model_dir):
    """
    功能：综合了前面所有函数，只要把参数传进去，就可以开始训练
    参数：
       x_train：训练集的特征，需要在 sess.run() 时，把特征传进去
       y_train：训练集的标签，需要在 sess.run() 时，把标签传进去
       x_eval：验证集的特征，需要在 sess.run() 时，把特征传进去
       y_eval：验证集的标签，需要在 sess.run() 时，把标签传进去
       time_step：时间阶段
       input_size：每一个阶段的输入维度
       n_classes：分类数
       batch_size：就是 batch_size
       lstm_layers_num：隐藏层数
       hidden_layer_units：隐藏层单元数
       train_epochs：训练多少个 epoch
       number_samples：训练集样本数量
       weight_decay：权重衰减系数，也就是 L2 正则化系数
       learning_rate：学习率
       keep_prob：Dropout 的保留率
       eval_cycle：验证的周期，每迭代多少次使用验证集验证一下，并且保存模型
       model_dir：模型保存的目录
    返回值：无
    """
    features = tf.placeholder(tf.float32, [None , time_step * input_size])  # [batch_size, 784]
    labels = tf.placeholder(tf.float32, [None , n_classes])  # [batch_size, n_classes]
    keep_prob_placeholder = tf.placeholder(tf.float32)  # 不被dropout的数据比例。在sess.run()时设置keep_prob具体的值
    global_step = tf.train.get_or_create_global_step()
    logits = forward_propagation(
        network_structure, features, time_step, input_size,
        hidden_layer_units, lstm_layers_num, n_classes, keep_prob_placeholder
    )
    train_op, error = backward_propagation(labels, logits, weight_decay, learning_rate, global_step)
    train(
        train_op, error, features, labels, logits, x_train, y_train, x_eval, y_eval, train_epochs, number_samples,
        batch_size, keep_prob_placeholder, keep_prob, epochs_per_eval, model_dir, global_step
    )
    save_hyper_parameter(
        model_dir, network_structure, time_step, input_size, hidden_layer_units, lstm_layers_num,
        n_classes, keep_prob, learning_rate, weight_decay, batch_size, train_epochs, epochs_per_eval
    )



def predict(model_dir, global_step_list, x_test, y_test=None, model_name="model.ckpt"):
    """
    功能：预测
    参数：
        Y_test：测试集的标签。如果在生产环境中，把 Y_test 设置为 None
        enable_moving_avg：是否启用滑动平均
    """
    # 加载模型超参数对象，然后获取模型的超参数
    hyper_parameter_path = os.path.join(model_dir, hyper_parameter_pkl_name)
    hyper_parameter_pkl_obj = load_pkl(hyper_parameter_path)
    network_structure = hyper_parameter_pkl_obj[network_structure_]
    time_step = hyper_parameter_pkl_obj[time_step_]
    input_size = hyper_parameter_pkl_obj[input_size_]
    hidden_layer_units = hyper_parameter_pkl_obj[hidden_layer_units_]
    lstm_layers_num = hyper_parameter_pkl_obj[lstm_layers_num_]
    n_classes = hyper_parameter_pkl_obj[n_classes_]
    # 新建一个计算图作为默认计算图，起一个别名叫 g
    with tf.Graph().as_default() as g:
        # 前向传播
        features = tf.placeholder(tf.float32, [None, time_step * input_size])  # [batch_size, 784]
        labels = tf.placeholder(tf.float32, [None, n_classes])  # [batch_size, n_classes]
        keep_prob_placeholder = tf.placeholder(tf.float32)  # 不被dropout的数据比例。在sess.run()时设置keep_prob具体的值
        global_step = tf.train.get_or_create_global_step()
        logits = forward_propagation(
            network_structure, features, time_step, input_size,
            hidden_layer_units, lstm_layers_num, n_classes, keep_prob_placeholder
        )
        if y_test is not None:
            hubers = tf.losses.huber_loss(labels=labels, predictions=logits)
            error = tf.reduce_sum(hubers)
        else:
            error = None
        # 对每一个模型进行预测
        y_pred_list = []  # 保存了每一个模型的预测值
        test_mse_list = []  # 保存了每一个模型的正确率
        saver = tf.train.Saver()
        for global_step in global_step_list:
            model_path = os.path.join(model_dir, model_name + "-" + str(global_step))
            with tf.Session() as sess:
                saver.restore(sess, model_path)  # 加载第 i 个模型
                # 前向传播得到预测值，类型是 numpy 数字矩阵
                y_pred_ndarray = sess.run(logits, feed_dict={features: x_test, keep_prob_placeholder: 1})
                y_pred_single_model = pd.DataFrame(y_pred_ndarray, columns=["ckpt-" + str(global_step)])  # 单个模型的预测值
                y_pred_list.append(y_pred_single_model)
                # 计算均方误差
                if y_test is not None:
                    test_error = sess.run(
                        error, feed_dict={features: x_test, labels: y_test, keep_prob_placeholder: 1}
                    )
                    print(str(global_step) + "   测试集：" + str(test_error))
                    test_mse_list.append(test_error)
        # # 通过多个模型融合计算最终的预测值
        y_pred_multi_model = pd.concat(y_pred_list, axis=1)  # 多个模型的预测值矩阵，每一列就是一个模型的预测值
        y_pred = y_pred_multi_model.mean(axis=1)  # 计算每一行唯一值有几个
        # 计算多个模型融合之后的平均正确率
        if y_test is not None:
            test_error = np.array(test_mse_list).mean()  # 平均正确率
            print("测试集：" + str(test_error))
        return y_pred



def load_pkl(pkl_Path):
    with open(pkl_Path , 'rb') as f:
        pkl_obj = pickle.load(f)
    return pkl_obj



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



def plot_results(y_true , y_pred , image_path , linewidth=2.5):
    plt.figure()
    num = y_true.shape[0]
    # plt.plot(list(range(num_train)) , X_train, color="b" , label="训练数据")
    plt.plot(list(range(num , num + len(y_true))) , y_true , color="b" , label="真实值" , linewidth=linewidth , marker="o")
    plt.plot(
        list(range(num, num + len(y_pred))), y_pred, color="r", label="预测值", linewidth=linewidth, linestyle="--" , marker="o"
    )
    plt.legend()
    # 中文字体设置
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.grid()  # 开启网格
    plt.tick_params(bottom="off", top="off", left="off", right="off")  # 把坐标上的齿去掉
    # 保存或者显示图片
    if image_path is not None:
        plt.savefig(image_path)
    else:
        plt.show()

