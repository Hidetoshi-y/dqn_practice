#!/bin/bash

pip install -U pip || exit 1
pip install numpy==1.17.4 || exit 1
pip install gym==0.15.4 || exit 1
pip install keras==2.3.1 || exit 1
pip install tensorflow==1.13.1 || exit 1
pip install keras-rl==0.4.2 || exit 1
pip install gym[atari] || exit 1