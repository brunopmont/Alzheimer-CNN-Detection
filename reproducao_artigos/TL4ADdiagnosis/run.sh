#!/bin/bash

DATA_DIR="3T_data_npy"
AUGMENTATION_DIR="augment"
SAVER_DIR_GA="GA_Results/ADnet"
SAVER_DIR_DA="DA_Results/ADnet"
MODEL="ADnet"
FEATURE_DIR="features"
AUGMENTATION="True" 

python GA_feature_extraction.py --model "$MODEL" \
                                --data_dir "$DATA_DIR" \
                                --saver_dir "$FEATURE_DIR"

python GA_classification.py --model "$MODEL" \
                            --feature_dir "$FEATURE_DIR" \
                            --augmentation "$AUGMENTATION" \
                            --saver_dir "$SAVER_DIR_GA"

python DA_data_augmentation.py --data_dir "$DATA_DIR" \
                               --saver_dir "$AUGMENTATION_DIR"

python DA_fine_tuning.py --model "$MODEL" \
                         --data_dir "$DATA_DIR" \
                         --augmentation_dir "$AUGMENTATION_DIR" \
                         --saver_dir "$SAVER_DIR_DA"