#!/bin/bash

bash utils/setup.sh

if [ -d 'data/labeled_dataset.csv' ];

then
    python train_lstm.py

else
     python fetch.py
     python train_lstm.py

fi