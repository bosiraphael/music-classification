import datetime
from data_loader import prepare_datasets, prepare_datasets_folds
from confusion_matrix import confusion_matrix
from train_CNN import train_CNN
from train_LSTM import train_LSTM
from train_ResNet import train_ResNet
import torch
import matplotlib.pyplot as plt
import numpy as np

X_lists, y_lists = prepare_datasets_folds()

results = np.zeros((5, 3))

now = datetime.datetime.now()
date = now.strftime("%Y_%m_%d_%H_%M")

#5-fold cross validation
for k in range(5):
    X_train = np.concatenate(X_lists[:k] + X_lists[k+1:])
    X_test = X_lists[k]
    y_train = np.concatenate(y_lists[:k] + y_lists[k+1:])
    y_test = y_lists[k]
    model_CNN, losses_CNN, losses_validation_CNN, train_accuracy_CNN, test_accuracy_CNN = train_CNN(X_train, X_test, None, y_train, y_test, None, 16, 200, 0.00024, 0.18)
    model_LSTM, losses_LSTM, losses_validation_LSTM, train_accuracy_LSTM, test_accuracy_LSTM = train_LSTM(X_train, X_test, None, y_train, y_test, None, 64, 200, 0.0005, 0.45)
    model_ResNet, losses_ResNet, losses_validation_ResNet, train_accuracy_ResNet, test_accuracy_ResNet = train_ResNet(X_train, X_test, None, y_train, y_test, None, 64, 10, 0.0001)
    results[k, 0] = test_accuracy_CNN[-1]
    results[k, 1] = test_accuracy_LSTM[-1]
    results[k, 2] = test_accuracy_ResNet[-1]

with open(f'comparison/results_{date}.txt', 'w') as f:
    f.write(f'CNN: {np.mean(results[:, 0])} +- {np.std(results[:, 0])}\n')
    f.write(f'LSTM: {np.mean(results[:, 1])} +- {np.std(results[:, 1])}\n')
    f.write(f'ResNet: {np.mean(results[:, 2])} +- {np.std(results[:, 2])}\n')