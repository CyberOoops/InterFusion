import csv
import numpy as np
import pandas as pd
import pickle as pkl


# preprocess for SWAT
train_path = "../../datasets/SWAT/SWaT_Dataset_Normal_v1.csv"
test_path = "../../datasets/SWAT/SWaT_Dataset_Attack_v0.csv"
with open(train_path, 'r')as file:
    csv_reader = csv.reader(file, delimiter=',')
    res_train = [row[1:-1] for row in csv_reader][2:]
    row_train = len(res_train)
    traindata = np.array(res_train, dtype=np.float32)

epsilo = 0.001
data_min = np.min(traindata, axis=0)
data_max = np.max(traindata, axis=0)+epsilo
for i in range(len(data_max)):
    if data_max[i] - data_min[i] < 10 * epsilo:
        data_min[i] = data_max[i]
        data_max[i] = 1 + data_max[i]

train = (traindata - data_min)/(data_max - data_min)
print("train shape ", train.shape)
pkl.dump(train, open('../data/processed/SWaT_train.pkl', 'wb'))
print('SWaT_train saved')

with open(test_path, 'r')as file:
    csv_reader = csv.reader(file, delimiter=',')
    res_test = [row[1:-1] for row in csv_reader][1:]
    row_test = len(res_test)
    testdata = np.array(res_test, dtype=np.float32)
    
rawdata = (testdata - data_min)/(data_max - data_min)
print("test shape ", rawdata.shape)
test = np.clip(rawdata, a_min=-1.0, a_max=3.0)
pkl.dump(test, open('../data/processed/SWaT_test.pkl', 'wb'))
print('SWaT_test saved')

with open(test_path, 'r')as file:
    csv_reader = csv.reader(file, delimiter=',')
    res = [row[-1]for row in csv_reader][1:]
    label_ = [0 if i == "Normal" else 1 for i in res]
labels = np.array(label_)
print("label shape ", labels.shape)
pkl.dump(labels, open('../data/processed/SWaT_test_label.pkl', 'wb'))
print('SWaT_test_label saved')



