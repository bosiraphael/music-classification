#!/bin/bash

BATCH_SIZE=100
EPOCHS=10
LR=0.0001

python3 train/train_ResNet.py -bs $BATCH_SIZE -e $EPOCHS --lr $LR