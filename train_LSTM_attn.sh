#!/bin/bash

set -e

dataset=$1

if [ $dataset = 'multiwoz' ]
then
    dataset='newAgentUser'
    data='newSpUser'
elif [ $dataset = 'dailydialog' ]
then
    dataset='newDailyDialog'
    data='DailyUser'
elif [ $dataset = 'ccpe' ]
then
    dataset='newCCPE'
    data='CCPEAgent'
else
    echo "wrong input."
    exit 1
fi

# for origin: LSTM + Global Attention
CUDA_VISIBLE_DEVICES=0 onmt_train -data data/$dataset/$data/agent -save_model nmt/$dataset/$data/agent-origin-model -world_size 1 -gpu_ranks 0
CUDA_VISIBLE_DEVICES=0 onmt_train -data data/$dataset/$data/user -save_model nmt/$dataset/$data/user-origin-model -world_size 1 -gpu_ranks 0
