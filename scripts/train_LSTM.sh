#!/bin/bash

BATCH_SIZE=64
EPOCHS=200
LR=0.0005

python3 train/train_LSTM.py -bs $BATCH_SIZE -e $EPOCHS --lr $LR