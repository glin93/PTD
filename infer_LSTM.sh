#!/bin/bash

set -e

dataset=$1
agent_model=$2
user_model=$3

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

for split in 'train' 'val' 'test'
do
    CUDA_VISIBLE_DEVICES=0 \
    onmt_translate \
    -model nmt/$dataset/$data/$agent_model \
    -src data/$dataset/$data/src-$split-agent.txt \
    -output data/$dataset/$data/pred-$split-agent-for-agent-woattn.txt \
    -gpu 0

    CUDA_VISIBLE_DEVICES=0 \
    onmt_translate \
    -model nmt/$dataset/$data/$agent_model \
    -src data/$dataset/$data/src-$split-user.txt \
    -output data/$dataset/$data/pred-$split-agent-for-user-woattn.txt \
    -gpu 0

    CUDA_VISIBLE_DEVICES=0 \
    onmt_translate \
    -model nmt/$dataset/$data/$user_model \
    -src data/$dataset/$data/src-$split-user.txt \
    -output data/$dataset/$data/pred-$split-user-for-user-woattn.txt \
    -gpu 0

    CUDA_VISIBLE_DEVICES=0 \
    onmt_translate \
    -model nmt/$dataset/$data/$user_model \
    -src data/$dataset/$data/src-$split-agent.txt \
    -output data/$dataset/$data/pred-$split-user-for-agent-woattn.txt \
    -gpu 0

done
