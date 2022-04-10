import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from data_loader import *
from pretrained_resnet import *
from sklearn.utils import shuffle
from pretrained_resnet import ResNet
import torchvision.transforms as transforms
from confusion_matrix import confusion_matrix

weight_decay = 1e-4

def batches(x, y, n, shuf = False):
    x_new = x.detach().clone()
    y_new = np.array(y.copy())
    b = []
    if shuf:
        x_new, y_new = shuffle(x_new, y_new)
    for i in range(0, x_new.shape[0], n):
        b.append((x_new[i:i + n], torch.from_numpy(y_new[i:i + n])))
    return b

def multi_channels(X):
    X_new = np.zeros((X.shape[0], 3, X.shape[1], X.shape[2]))
    for k in range(3):
        X_new[:, k, :, :] = X
    return X_new

def train_ResNet(X_train, X_test, X_validation, y_train, y_test, y_validation, batch_size = 100, epochs = 100, lr = 0.05):
    # we use GPU if available, otherwise CPU
    torch.set_default_dtype(torch.double)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    transf = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    X_train = transf(torch.from_numpy(multi_channels(X_train)))
    X_test = transf(torch.from_numpy(multi_channels(X_test)))
    if X_validation is not None:
        X_validation = transf(torch.from_numpy(multi_channels(X_validation)))

    model = ResNet()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device) # puts model on GPU / CPU

    if torch.cuda.device_count() > 1:
        model.module.freeze()
        # optimization hyperparameters
        optimizer = torch.optim.Adam(model.module.parameters(), lr = lr, weight_decay=weight_decay)
    else:
        model.freeze()
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss()

    losses = []
    losses_validation = []
    train_accuracy = []
    test_accuracy = []

    train_accuracy = []
    test_accuracy = []

    # main loop (train+test)
    for epoch in range(epochs):
        if epoch%5 == 0 and epoch != 0:
            if epoch%10 == 0:
                model.freeze()
            else:
                model.unfreeze()

        # training
        if torch.cuda.device_count() > 1:
            model.module.train() # mode "train" agit sur "dropout" ou "batchnorm"
        else:
            model.train()
        losses_epoch = []
        b = batches(X_train, y_train, batch_size, True)
        for batch_idx, (x, target) in enumerate(b):
            optimizer.zero_grad()
            
            x, target = x.to(device), target.to(device)
            
            out = model(x)
            
            #print(out.shape)
            loss = loss_fn(out, target)
            loss.backward()
            grad_clip = 0.1
            if grad_clip: 
                if torch.cuda.device_count() > 1:
                    nn.utils.clip_grad_value_(model.module.parameters(), grad_clip)  
                else:
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()

            losses_epoch.append(loss.item())
            if batch_idx %10 ==0:
                print('epoch {} batch {} [{}/{}] training loss: {}'.format(epoch,batch_idx,batch_idx*len(x),
                        len(y_train),loss.item()))

        losses.append(np.mean(losses_epoch))

        # testing
        if torch.cuda.device_count() > 1:
            model.module.eval()         
        else:
            model.eval()

        #train accuracy
        correct = 0
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(b):
                

                x, target = x.to(device), target.to(device)
                
                out = model(x)


                loss = loss_fn(out, target)
                # _, prediction = torch.max(out.data, 1)
                prediction = out.argmax(dim=1, keepdim=True) # index of the max log-probability
                correct += prediction.eq(target.view_as(prediction)).sum().item()
        taux_classif_train = 100. * correct / len(y_train)

        train_accuracy.append(taux_classif_train)

        losses_batches = []

        #validation
        if X_validation is not None:
            with torch.no_grad():
                b = batches(X_validation, y_validation, batch_size)
                for batch_idx, (x, target) in enumerate(b):
                    

                    x, target = x.to(device), target.to(device)
                    
                    out = model(x)

                    losses_batches.append(loss_fn(out, target).cpu().detach().numpy())

            losses_validation.append(np.mean(losses_batches))

        #test accuracy
        correct = 0
        with torch.no_grad():
            b = batches(X_test, y_test, batch_size)
            for batch_idx, (x, target) in enumerate(b):
                x, target = x.to(device), target.to(device)
                out = model(x)

                loss = loss_fn(out, target)
                # _, prediction = torch.max(out.data, 1)
                prediction = out.argmax(dim=1, keepdim=True) # index of the max log-probability
                correct += prediction.eq(target.view_as(prediction)).sum().item()
        taux_classif_test = 100. * correct / len(y_test)

        test_accuracy.append(taux_classif_test)
        print('Accuracy: {}/{} (tx {:.2f}%, err {:.2f}%)\n'.format(correct,
        len(y_test), taux_classif_test, 100.-taux_classif_test))
   
    return model, losses, losses_validation, train_accuracy, test_accuracy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, help="batch size")
    parser.add_argument("-e", "--epochs", type=int, help="epochs")
    parser.add_argument("--lr", type=float, help="learning rate")
    args = parser.parse_args()

    now = datetime.datetime.now()
    date = now.strftime("%Y_%m_%d_%H_%M")
    X_train, X_test, X_validation, y_train, y_test, y_validation = prepare_datasets(0.2, 0.1)


    model, losses, losses_validation, train_accuracy, test_accuracy = train_ResNet(X_train, X_test, X_validation, y_train, y_test, y_validation, args.batch_size, args.epochs, args.lr)

    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), 'trained_models/models/ResNet_' + date + \
        '_epochs_' + str(args.epochs) + \
        '_batch_size_' + str(args.batch_size) + \
        '_lr_' + str(args.lr))
    else:
        torch.save(model.state_dict(), 'trained_models/models/ResNet_' + date + \
        '_epochs_' + str(args.epochs) + \
        '_batch_size_' + str(args.batch_size) + \
        '_lr_' + str(args.lr))
    
    
    fig, axs = plt.subplots(2, 2, constrained_layout = True)
    axs[0, 0].plot(losses)
    axs[0, 0].set_title('Train Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 1].plot(losses_validation)
    axs[0, 1].set_title('Validation Loss')
    axs[0, 1].set_xlabel('Epochs')
    axs[1, 0].plot(train_accuracy)
    axs[1, 0].set_title('Train Accuracy')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 1].plot(test_accuracy)
    axs[1, 1].set_title('Test Accuracy')
    axs[1, 1].set_xlabel('Epochs')
    fig.savefig('trained_models/results/ResNet_' + date + \
        '_epochs_' + str(args.epochs) + \
        '_batch_size_' + str(args.batch_size) + \
        '_lr_' + str(args.lr) + '.png')
