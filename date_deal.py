import csv
import csv
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
# 定义CSV文件路径
def normalize_rows(data):
    # 计算每一行的最小值和最大值
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)

    # 计算每一行的范围
    ranges = max_vals - min_vals

    # 对每一行进行归一化
    normalized_data = (data - min_vals) / ranges

    return normalized_data

def standardsize(data):
    # 创建StandardScaler对象
    scaler = StandardScaler()
    data_normal = scaler.fit_transform(data)
    return data_normal



def softmax(x):
    # 对输入数组进行指数运算
    exp_x = np.exp(x)

    # 对每一行进行求和，用于计算分母部分
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)

    # 计算Softmax函数的结果
    softmax_result = exp_x / sum_exp_x

    return softmax_result

def add_one_to_zeros(matrix):
    # 使用numpy中的where函数找到所有等于0的位置，并将它们加一
    matrix[np.where(matrix == 0)] += 1.
    return matrix

def data_read(csv_file_path,batch_size):
    # 创建一个空列表用于存储CSV文件中的数据
    data = []
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        # 跳过表头（如果有的话）
        header = next(csv_reader)
        # 逐行读取数据并添加第五列及之后的数据到data列表中
        for row in csv_reader:
            data.append(np.array(row[5:]).astype(np.float32))
    # 数据的预处理
    data = add_one_to_zeros(np.array(data))
    data = np.log10(np.abs(np.array(data)))

    train_data = data[:, :70]
    train_label = data[:, 70:]

    # train_data = np.log10(np.abs(np.array(train_data)))
    train_data = standardsize(train_data)

    X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_label, random_state=600, shuffle=True)

    datas_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size)
    train_dataset = datas_train.shuffle(100)

    test_train = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(batch_size)
    test_dataset = test_train.shuffle(100)

    return train_dataset,test_dataset

# csv_file_path = r'D:\pj\co2\data\train.csv'
# train_dataset, test_dataset = data_read(csv_file_path=csv_file_path,batch_size=16)
# data_train = next(iter(train_dataset))
# data_test = next(iter(test_dataset))
#
# print()