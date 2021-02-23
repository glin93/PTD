#!/bin/bash

set -e


data=$1

if [ $data = 'multiwoz' ]
then
    data='newSpUser'
elif [ $data = 'dailydialog' ]
then
    data='DailyUser'
elif [ $data = 'ccpe' ]
then
    data='CCPEAgent'
else
    echo "wrong input."
    exit 1
fi

if [ $data = 'newSpUser' ]
then
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -u trainBERT.py \
        --model_type bert \
        --model_name_or_path bert-base-cased \
        --task_name his \
        --do_train \
        --do_test \
        --do_lower_case \
        --evaluate_during_training \
        --data_dir data/newAgentUser/newSpUser \
        --output_dir bert/newAgentUser/newSpUser/his \
        --overwrite_output_dir \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 32 \
        --per_gpu_eval_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --save_steps 100 \
        --logging_steps 100
else
    if [ $data = 'DailyUser' ]
    then
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        python -u trainBERT.py \
        --model_type bert \
        --model_name_or_path bert-base-cased \
        --task_name his \
        --do_train \
        --do_test \
        --do_lower_case \
        --evaluate_during_training \
        --data_dir data/newDailyDialog/DailyUser \
        --output_dir bert/newDailyDialog/DailyUser/his \
        --overwrite_output_dir \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 32 \
        --per_gpu_eval_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --save_steps 100 \
        --logging_steps 100
    else # CCPEAgent
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        python -u trainBERT.py \
        --model_type bert \
        --model_name_or_path bert-base-cased \
        --task_name his \
        --do_train \
        --do_test \
        --do_lower_case \
        --evaluate_during_training \
        --data_dir data/newCCPE/CCPEAgent \
        --output_dir bert/newCCPE/CCPEAgent/his \
        --overwrite_output_dir \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 16 \
        --per_gpu_eval_batch_size 16 \
        --learning_rate 5e-5 \
        --num_train_epochs 4.0 \
        --save_steps 100 \
        --logging_steps 100
    fi
fi
