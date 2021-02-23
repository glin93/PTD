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

# for woattn: LSTM

CUDA_VISIBLE_DEVICES=0 onmt_train -data data/$dataset/$data/agent  -save_model nmt/$dataset/$data/agent-woattn-model -world_size 1 -gpu_ranks 0 --global_attention none
CUDA_VISIBLE_DEVICES=0 onmt_train -data data/$dataset/$data/user -save_model nmt/$dataset/$data/user-woattn-model -world_size 1 -gpu_ranks 0 --global_attention none