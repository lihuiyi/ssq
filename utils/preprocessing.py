# -*- coding: utf-8 -*-

import os
import collections
import pandas as pd
from sklearn.preprocessing import OneHotEncoder #哑编码
from sklearn.preprocessing import StandardScaler #标准化
from sklearn.preprocessing import Normalizer #归一化
from sklearn.preprocessing import Binarizer #二值化
import pickle




class OneHotEncoder_:
    """
    功能：哑编码
    """
    def __init__(self):
        self.columns = None
        self.oneHotEncoder = OneHotEncoder(handle_unknown='ignore')


    def fit_transform(self, dataset, columns):
        """
        功能：训练集哑编码
        参数：
            dataset：训练集，DataFrame 类型
            columns：需要哑编码的列名集合，list 类型
        返回值：哑编码后的数据，DataFrame 类型
        """
        if type(columns) is str:
            columns = [columns]
        # 保存 columns, 以备下次使用
        self.columns = columns
        # 进行哑编码
        data = self.oneHotEncoder.fit_transform(dataset.loc[:, columns]).toarray()  # 进行哑编码
        classes_list = self.oneHotEncoder.categories_  # 每一列中不重复的类别值
        # 构造出新的列名
        new_columns = self.get_new_columns(classes_list)
        # 把哑编码后的数据转换为 DataFrame 格式
        data = pd.DataFrame(data, columns=new_columns)
        return data



    # 测试集哑编码
    # 参数：数据集、列名
    def transform(self, dataset):
        """
        功能：测试集哑编码
        参数：
            dataset：测试集，DataFrame 类型
        返回值：哑编码后的数据，DataFrame 类型
        """
        # 进行哑编码
        data = self.oneHotEncoder.transform(dataset.loc[:, self.columns]).toarray()  # 进行哑编码
        classes_list = self.oneHotEncoder.categories_  # 每一列中不重复的类别值
        # 构造出新的列名
        new_columns = self.get_new_columns(classes_list)
        # 把哑编码后的数据转换为 DataFrame 格式
        data = pd.DataFrame(data, columns=new_columns)
        return data



    def get_new_columns(self, classes_list):
        """
        功能：构造出哑编码后的新列名
        参数：
            classes_list：测试集，DataFrame 类型
        返回值：哑编码后的数据，DataFrame 类型
        """
        new_columns = []  # 用于保存哑编码后的新列名
        for i in range(len(self.columns)):
            for j in range(len(classes_list[i])):
                new_columns.append(self.columns[i] + "_" + str(classes_list[i][j]))
        return new_columns




class StandardScaler_:
    """
    功能：标准化
    """
    def __init__(self):
        self.columns = None
        self.standardScaler = StandardScaler()


    def fit_transform(self, dataset, columns):
        """
        # 功能：训练集标准化
        参数：
            dataset：训练集，DataFrame 类型
            columns：需要标准化的列名集合，list 类型
        返回值：标准化后的数据，DataFrame 类型
        """
        if type(columns) is str:
            columns = [columns]
        # 保存 columns, 以备下次使用
        if type(columns) is list:
            self.columns = columns
        else:
            self.columns = columns.tolist()
        # 标准化
        data = self.standardScaler.fit_transform(dataset.loc[:, columns])
        data = pd.DataFrame(data, columns=columns)
        return data



    def transform(self, dataset):
        """
        # 功能：测试集标准化
        参数：
            dataset：测试集，DataFrame 类型
        返回值：标准化后的数据，DataFrame 类型
        """
        # 标准化
        data = self.standardScaler.transform(dataset.loc[:, self.columns])
        data = pd.DataFrame(data, columns=self.columns)
        return data




class Normalizer_:
    """
    功能：归一化
    归一化与标准化的区别：
        标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，将样本的特征值转换到同一量纲下。
        归一化是依照特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，
        拥有统一的标准，也就是说都转化为“单位向量”。
    """
    def __init__(self):
        self.columns = None
        self.normalizer = Normalizer()


    def fit_transform(self, dataset, columns):
        if type(columns) is str:
            columns = [columns]
        # 保存 columns, 以备下次使用
        if type(columns) is list:
            self.columns = columns
        else:
            self.columns = columns.tolist()
        # 归一化
        data = self.normalizer.fit_transform(dataset.loc[:, columns])  # 标准化后，返回矩阵格式的数据
        data = pd.DataFrame(data, columns=columns)
        return data



    def transform(self, dataset):
        # 标准化
        data = self.normalizer.transform(dataset.loc[:, self.columns])
        data = pd.DataFrame(data, columns=self.columns)
        return data




class Binarizer_:
    """
    功能：二值化。小于阈值的为0，大于阈值的为1
    """
    def __init__(self, threshold=None):
        self.columns = None  # 阈值。小于阈值的为0，大于阈值的为1
        self.binarizer = Binarizer(threshold=threshold)


    def fit_transform(self, dataSet, columns):
        if type(columns) is str:
            columns = [columns]
        # 保存 columns, 以备下次使用
        if type(columns) is list:
            self.columns = columns
        else:
            self.columns = columns.tolist()
        data = self.binarizer.fit_transform(dataSet.loc[:, columns])  # 标准化后，返回矩阵格式的数据
        data = pd.DataFrame(data, columns=columns)
        return data



    def transform(self, dataset):
        # 标准化
        data = self.binarizer.transform(dataset.loc[:, self.columns])
        data = pd.DataFrame(data, columns=self.columns)
        return data




class Data_preprocessing():
    """
    功能：数据预处理综合
    """
    def __init__(self):
        self.preprocessing_obj_dict = collections.OrderedDict()


    def fit_transform(self, dataset, features_onehot_columns, features_standardScaler_columns,
                      features_normalizer_columns, label_name, label_preprocessing_method, pkl_dir):
        data_list = []  # 用于保存预处理后的数据
        # 特征哑编码
        if features_onehot_columns is not None:
            onehot = OneHotEncoder_()
            onehot_feature = onehot.fit_transform(dataset, features_onehot_columns)
            data_list.append(onehot_feature)
            self.preprocessing_obj_dict["OneHotEncoder"] = onehot
        # 特征标准化
        if features_standardScaler_columns is not None:
            standardScaler = StandardScaler_()
            standardScaler_feature = standardScaler.fit_transform(dataset, features_standardScaler_columns)
            data_list.append(standardScaler_feature)
            self.preprocessing_obj_dict["StandardScaler"] = standardScaler
        # 特征归一化
        if features_normalizer_columns is not None:
            normalizer = Normalizer_()
            normalizer_feature = normalizer.fit_transform(dataset, features_normalizer_columns)
            data_list.append(normalizer_feature)
            self.preprocessing_obj_dict["Normalizer"] = normalizer
        # 标签预处理
        # if label_preprocessing_method is not None:
        label = self.label_preprocessing(dataset, label_name, label_preprocessing_method)
        data_list.append(label)
        # 把全部预处理数据连接起来
        data = pd.concat(data_list, axis=1)
        # 保存预处理对象
        self.save_pkl(pkl_dir)
        return data



    def transform(self, dataset, pkl_dir):
        preprocessing_obj_dict = self.load_pkl(pkl_dir)
        data_list = []
        for key in preprocessing_obj_dict.keys():
            preprocessing_obj = preprocessing_obj_dict[key]
            data = preprocessing_obj.transform(dataset)
            data_list.append(data)
        data = pd.concat(data_list, axis=1)
        return data



    def label_preprocessing(self, dataset, label_name, label_preprocessing_method):
        """
        功能：标签预处理
        参数：
            dataset：数据集
            label_name：标签名称
            label_preprocessing_method：标签预处理的方式。有2个可选值，"Onehot"、"StandardScaler"
        """
        label_preprocessing_obj = None
        if label_preprocessing_method == "Onehot":
            label_preprocessing_obj = OneHotEncoder_()
        elif label_preprocessing_method == "StandardScaler":
            label_preprocessing_obj = StandardScaler_()
        elif label_preprocessing_method is None:
            self.preprocessing_obj_dict["label_preprocessing_obj"] = label_preprocessing_obj
            return dataset.loc[:, [label_name]]
        # 标签预处理
        label = label_preprocessing_obj.fit_transform(dataset, label_name)
        # 把 label_preprocessing_obj 保存到 preprocessing_obj_dict 字典中
        self.preprocessing_obj_dict["label_preprocessing_obj"] = label_preprocessing_obj
        return label



    def save_pkl(self, pkl_dir, pkl_name="数据预处理.pkl"):
        """
        功能：把 self.preprocessing_obj_dict 预处理字典保存到 pkl 文件
        参数：
            pkl_dir：pkl 文件的输出目录
            pkl_name：pkl 文件的名称
        返回值：无
        """
        is_exists = os.path.exists(pkl_dir)  # 判断一个目录是否存在
        if is_exists is False:
            os.makedirs(pkl_dir)  # 创建目录
        processing_pkl_Path = os.path.join(pkl_dir, pkl_name)
        with open(processing_pkl_Path, 'wb') as f:
            pickle.dump(self.preprocessing_obj_dict, f)



    def load_pkl(self, pkl_dir, pkl_name="数据预处理.pkl"):
        """
        功能：从 pkl 文件中加载 self.preprocessing_obj_dict 预处理字典
        参数：
            pkl_dir：pkl 文件的目录
        返回值：无
        """
        pkl_Path = os.path.join(pkl_dir, pkl_name)
        with open(pkl_Path, 'rb') as f:
            pkl_obj = pickle.load(f)
        return pkl_obj

