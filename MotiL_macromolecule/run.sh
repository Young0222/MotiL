#!/bin/bash

# For the GO dataset
python go.py --level 'cc' --gpu 6 --seed 0 --lr 1e-3 --batch-size 64 --wd 5e-4 --num-epochs 300 \
--base-width 32 --kernel-channels 24 --lr-milestone 300 400

# For the EC dataset
python ec.py --gpu 4 --seed 2027 --batch-size 24 --num-pretrain-epochs 0 --ckpt-path './ckpt'