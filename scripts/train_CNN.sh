#!/bin/bash

BATCH_SIZE=16
EPOCHS=200
LR=0.00024

python3 train/train_CNN.py -bs $BATCH_SIZE -e $EPOCHS --lr $LR