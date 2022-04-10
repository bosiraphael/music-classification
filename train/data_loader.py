import numpy as np
import json
import random as rd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_data(data_path,with_spectro=False):
    
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    z = np.array(data['mapping'])
    if with_spectro:
        zz = np.array(data["spectro"])
        return X, y, z, zz
    return X,y,z

def prepare_datasets(test_size, validation_size,with_spectro=False):
    
    # load data

    if with_spectro:
        X, y, z, zz = load_data("data/data_11.json",True)
        print(np.shape(X))
        print(np.shape(zz))
        newX=np.zeros((np.shape(X)[0],np.shape(X[0])[0],np.shape(X[0])[1]+np.shape(np.transpose(zz[0]))[1]))
        for i in range(np.shape(X)[0]):
            newX[i,:,0:np.shape(X[0])[1]]=X[i]
            newX[i,:,np.shape(X[0])[1]:np.shape(X[0])[1]+np.shape(np.transpose(zz[0]))[1]]=np.transpose(zz[1])
        X=newX.copy()
    else:
        X, y, z = load_data("data/data_10.json")
    # create train, validation and test split
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    X_validation = []
    y_validation  = []
    tirage=[i for i in range(10)]
    rd.shuffle(tirage)
    list_y=[0]*10
    for i in range(len(y)):
        list_y[y[i]]+=1
    #print(list_y)
    new_X=[]
    new_y=[]
    for i in range(len(y)//6):
        sous_new=[]
        sous_new_y=[]
        for k in range(6):
            sous_new.append(X[i*6+k])
            sous_new_y.append(y[i*6+k])
        new_X.append(sous_new)
        new_y.append(sous_new_y)
    #print(len(new_X))
    new_X, new_y = shuffle(new_X, new_y)

    X_train1=new_X[0:int((1-test_size-validation_size)*len(new_X))]
    X_test1=new_X[int((1-test_size-validation_size)*len(new_X))+1:int((1-validation_size)*len(new_X))]
    X_validation1=new_X[int((1-validation_size)*len(new_X))+1:len(new_X)]
    
    y_train1=new_y[0:int((1-test_size-validation_size)*len(new_y))]
    y_test1=new_y[int((1-test_size-validation_size)*len(new_y))+1:int((1-validation_size)*len(new_y))]
    y_validation1=new_y[int((1-validation_size)*len(new_y))+1:len(new_y)]

    for i in range(len(X_train1)):
        X_train=X_train+X_train1[i]
        y_train=y_train+y_train1[i]
    for i in range(len(X_test1)):
        X_test=X_test+X_test1[i]
        y_test=y_test+y_test1[i]
    for i in range(len(X_validation1)):
        X_validation=X_validation+X_validation1[i]
        y_validation=y_validation+y_validation1[i]
    
    X_train,X_test,X_validation=np.array(X_train),np.array(X_test),np.array(X_validation)
    print("X_train ",np.shape(X_train))
    print("X_test ",np.shape(X_test))
    print("X_validation ",np.shape(X_validation))
    return X_train,X_test,X_validation,y_train,y_test,y_validation

def prepare_datasets_shuffle(test_size, validation_size):
    
    # load data
    X, y, z = load_data("data/data_10.json")
    # create train, validation and test split
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    X_validation = []
    y_validation  = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size+validation_size, stratify=y)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=validation_size/test_size, stratify=y_test)
    
    return X_train,X_test,X_validation,y_train,y_test,y_validation
        
# le data path est : "data/data_10.json"
#prepare_datasets(0.2,0.2,True)


def prepare_datasets_folds():
    
    # load data
    X, y, z = load_data("data/data_10.json")
    print('y',len(y))
    # create 5 folds
    X_train_1 = []
    X_train_2 = []
    X_train_3 = []
    X_train_4 = []
    X_train_5 = []
    y_train_1 = []
    y_train_2 = []
    y_train_3 = []
    y_train_4 = []
    y_train_5 = []
    
    tirage=[i for i in range(10)]
    rd.shuffle(tirage)
    list_y=[0]*10
    for i in range(len(y)):
        list_y[y[i]]+=1
    #print(list_y)
    new_X=[]
    new_y=[]
    for i in range(len(y)//6):
        sous_new=[]
        sous_new_y=[]
        for k in range(6):
            sous_new.append(X[i*6+k])
            sous_new_y.append(y[i*6+k])
        new_X.append(sous_new)
        new_y.append(sous_new_y)
    #print(len(new_X))
    new_X, new_y = shuffle(new_X, new_y)
    print('new_y',len(new_y))

    X_train1=new_X[0:int(0.2*len(new_X))]
    X_train2=new_X[int(0.2*len(new_X)):int(0.4*len(new_X))]
    X_train3=new_X[int(0.4*len(new_X)):int(0.6*len(new_X))]
    X_train4=new_X[int(0.6*len(new_X)):int(0.8*len(new_X))]
    X_train5=new_X[int(0.8*len(new_X)):len(new_X)]

    y_train1=new_y[0:int(0.2*len(new_y))]
    y_train2=new_y[int(0.2*len(new_y)):int(0.4*len(new_y))]
    y_train3=new_y[int(0.4*len(new_y)):int(0.6*len(new_y))]
    y_train4=new_y[int(0.6*len(new_y)):int(0.8*len(new_y))]
    y_train5=new_y[int(0.8*len(new_y)):len(new_y)]
    
    print("y_train1 ",len(y_train1))
    print("y_train2 ",len(y_train2))
    print("y_train3 ",len(y_train3))
    print("y_train4 ",len(y_train4))
    print("y_train5 ",len(y_train5))
    for i in range(len(X_train1)):
        X_train_1 = X_train_1 + X_train1[i]
        y_train_1 = y_train_1 + y_train1[i]
    for i in range(len(X_train2)):
        X_train_2 = X_train_2 + X_train2[i]
        y_train_2 = y_train_2 + y_train2[i]
    for i in range(len(X_train3)):
        X_train_3 = X_train_3 + X_train3[i]
        y_train_3 = y_train_3 + y_train3[i]
    for i in range(len(X_train4)):
        X_train_4 = X_train_4 + X_train4[i]
        y_train_4 = y_train_4 + y_train4[i]
    for i in range(len(X_train5)):
            X_train_5 = X_train_5 + X_train5[i]
            y_train_5 = y_train_5 + y_train5[i]

    X_train_1 = np.array(X_train_1)
    X_train_2 = np.array(X_train_2)
    X_train_3 = np.array(X_train_3)
    X_train_4 = np.array(X_train_4)
    X_train_5 = np.array(X_train_5)

    print("X_train_1 ",np.shape(X_train_1))
    print("X_train_2 ",np.shape(X_train_2))
    print("X_train_3 ",np.shape(X_train_3))
    print("X_train_4 ",np.shape(X_train_4))
    print("X_train_5 ",np.shape(X_train_5))
    print("y_train_1 ",len(y_train_1))
    print("y_train_2 ",len(y_train_2))
    print("y_train_3 ",len(y_train_3))
    print("y_train_4 ",len(y_train_4))
    print("y_train_5 ",len(y_train_5))

    return [X_train_1,X_train_2,X_train_3,X_train_4,X_train_5],[y_train_1,y_train_2,y_train_3,y_train_4,y_train_5]


#Xs, ys =prepare_datasets_folds()