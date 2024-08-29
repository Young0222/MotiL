#!/bin/bash

# For the GO dataset
python go.py --pretrain 'True' --gpu 6 --level 'cc' --batch-size 64 --lr 1e-3  --wd 5e-4 --num-epochs 300 \
--base-width 32 --kernel-channels 24 --lr-milestone 300 400

# For the EC dataset
python ec.py --pretrain 'True' --gpu 6 --batch-size 24