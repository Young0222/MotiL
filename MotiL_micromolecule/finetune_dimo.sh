# BBBP
 python train.py \
    --data_path ./data/bbbp.csv \
    --exp_id bbbp \
    --dataset_type classification \
    --epochs 10 \
    --num_runs 3 \
    --gpu 5 \
    --batch_size 16 \
    --seed 2024 \
    --init_lr 1e-5  \
    --final_lr 5e-6 \
    --wd 5e-5 \
    --warmup_epochs 0.0 \
    --split_type 'scaffold_balanced' \
    --exp_name finetune \
    --split_sizes 0.8 0.1 0.1 \
    --step 'finetune' \
    --checkpoint_path "./dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl" 

# ClinTox
python train.py \
    --data_path ./data/clintox.csv \
    --exp_id clintox \
    --dataset_type classification \
    --epochs 10 \
    --num_runs 3 \
    --gpu 5 \
    --batch_size 16 \
    --seed 2024 \
    --init_lr 1e-5  \
    --final_lr 5e-6 \
    --wd 5e-5 \
    --warmup_epochs 0.0 \
    --split_type 'scaffold_balanced' \
    --exp_name finetune \
    --split_sizes 0.8 0.1 0.1 \
    --step 'finetune' \
    --checkpoint_path "./dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl"

# BACE
python train.py \
    --data_path ./data/bace.csv \
    --exp_id bace \
    --dataset_type classification \
    --epochs 20 \
    --num_runs 3 \
    --gpu 5 \
    --batch_size 16 \
    --seed 2024 \
    --init_lr 1e-3  \
    --final_lr 5e-4 \
    --wd 1e-5 \
    --warmup_epochs 0.0 \
    --split_type 'scaffold_balanced' \
    --exp_name finetune \
    --split_sizes 0.8 0.1 0.1 \
    --step 'finetune' \
    --checkpoint_path "./dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl"

# ESOL
python train.py \
    --data_path ./data/esol.csv \
    --dataset_type regression \
    --epochs 100 \
    --num_runs 3 \
    --gpu 5 \
    --batch_size 64 \
    --seed 2024 \
    --init_lr 1e-3  \
    --final_lr 1e-6 \
    --wd 1e-5 \
    --warmup_epochs 0.0 \
    --split_type 'scaffold_balanced' \
    --exp_name scaffold_balanced \
    --exp_id esol \
    --metric rmse \
    --checkpoint_path "./dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl"