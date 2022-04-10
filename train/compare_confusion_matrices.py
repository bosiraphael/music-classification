import datetime
from data_loader import prepare_datasets, prepare_datasets_shuffle
from confusion_matrix import confusion_matrix
from train_CNN import train_CNN
from train_LSTM import train_LSTM
from train_ResNet import train_ResNet
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from train_ResNet import multi_channels

now = datetime.datetime.now()
date = now.strftime("%Y_%m_%d_%H_%M")

X_train, X_test, X_validation, y_train, y_test, y_validation = prepare_datasets_shuffle(0.2, 0.1)

model_ResNet, losses_ResNet, losses_validation_ResNet, train_accuracy_ResNet, test_accuracy_ResNet = train_ResNet(X_train, X_test, X_validation, y_train, y_test, y_validation, 64, 30, 0.0001)
model_CNN, losses_CNN, losses_validation_CNN, train_accuracy_CNN, test_accuracy_CNN = train_CNN(X_train, X_test, X_validation, y_train, y_test, y_validation, 16, 200, 0.00024, 0.2, 20)
model_LSTM, losses_LSTM, losses_validation_LSTM, train_accuracy_LSTM, test_accuracy_LSTM = train_LSTM(X_train, X_test, X_validation, y_train, y_test, y_validation, 64, 100, 0.00048, 0.2, 20)

torch.save(model_CNN.state_dict(), 'comparison/models/CNN_' + date)
torch.save(model_LSTM.state_dict(), 'comparison/models/LSTM_' + date)
torch.save(model_ResNet.state_dict(), 'comparison/models/ResNet_' + date)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transf_1 = transforms.Normalize(
        mean=[0.5],
        std=[0.25],
    )

transf_2 = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

X_test_CNN_LSTM = transf_1(torch.from_numpy(X_test))
X_test_CNN_LSTM = X_test_CNN_LSTM.to(device)
X_test_ResNet = transf_2(torch.from_numpy(multi_channels(X_test)))
X_test_ResNet = X_test_ResNet.to(device)
y_test = torch.from_numpy(np.array(y_test))
y_test = y_test.to(device)

fig, axs = plt.subplots(2, 2, constrained_layout = True)
axs[0, 0].plot(losses_CNN)
axs[0, 0].set_title('Train Loss')
axs[0, 0].set_xlabel('Epochs')
axs[0, 1].plot(losses_validation_CNN)
axs[0, 1].set_title('Validation Loss')
axs[0, 1].set_xlabel('Epochs')
axs[1, 0].plot(train_accuracy_CNN)
axs[1, 0].set_title('Train Accuracy')
axs[1, 0].set_xlabel('Epochs')
axs[1, 1].plot(test_accuracy_CNN)
axs[1, 1].set_title('Test Accuracy')
axs[1, 1].set_xlabel('Epochs')
fig.savefig('comparison/results/plots/CNN_' + date + '_shuffle.png')

confusion_matrix(model_CNN, X_test_CNN_LSTM, y_test, f'comparison/results/confusion_matrices/CNN_{date}_shuffle.png', 'CNN')

fig, axs = plt.subplots(2, 2, constrained_layout = True)
axs[0, 0].plot(losses_LSTM)
axs[0, 0].set_title('Train Loss')
axs[0, 0].set_xlabel('Epochs')
axs[0, 1].plot(losses_validation_LSTM)
axs[0, 1].set_title('Validation Loss')
axs[0, 1].set_xlabel('Epochs')
axs[1, 0].plot(train_accuracy_LSTM)
axs[1, 0].set_title('Train Accuracy')
axs[1, 0].set_xlabel('Epochs')
axs[1, 1].plot(test_accuracy_LSTM)
axs[1, 1].set_title('Test Accuracy')
axs[1, 1].set_xlabel('Epochs')
fig.savefig('comparison/results/plots/LSTM_' + date + '_shuffle.png')

confusion_matrix(model_LSTM, X_test_CNN_LSTM, y_test, f'comparison/results/confusion_matrices/LSTM_{date}_shuffle.png')

fig, axs = plt.subplots(2, 2, constrained_layout = True)
axs[0, 0].plot(losses_ResNet)
axs[0, 0].set_title('Train Loss')
axs[0, 0].set_xlabel('Epochs')
axs[0, 1].plot(losses_validation_ResNet)
axs[0, 1].set_title('Validation Loss')
axs[0, 1].set_xlabel('Epochs')
axs[1, 0].plot(train_accuracy_ResNet)
axs[1, 0].set_title('Train Accuracy')
axs[1, 0].set_xlabel('Epochs')
axs[1, 1].plot(test_accuracy_ResNet)
axs[1, 1].set_title('Test Accuracy')
axs[1, 1].set_xlabel('Epochs')
fig.savefig('comparison/results/plots/ResNet_' + date + '_shuffle.png')

confusion_matrix(model_ResNet, X_test_ResNet, y_test, f'comparison/results/confusion_matrices/ResNet_{date}_shuffle.png')