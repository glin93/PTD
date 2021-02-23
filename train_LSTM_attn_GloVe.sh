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

# prepare glove
python tools/embeddings_to_torch.py -emb_file_both "data/glove.6B.300d.txt" \
-dict_file "data/$dataset/$data/agent.vocab.pt" \
-output_file "data/$dataset/$data/agent.embeddings"

python tools/embeddings_to_torch.py -emb_file_both "data/glove.6B.300d.txt" \
-dict_file "data/$dataset/$data/user.vocab.pt" \
-output_file "data/$dataset/$data/user.embeddings"


# for glove: LSTM + Attention + GloVe
CUDA_VISIBLE_DEVICES=0 \
onmt_train \
-word_vec_size 300 \
-pre_word_vecs_enc "data/$dataset/$data/agent.embeddings.enc.pt" \
-pre_word_vecs_dec "data/$dataset/$data/agent.embeddings.dec.pt" \
-data data/$dataset/$data/agent \
-save_model nmt/$dataset/$data/agent-glove-model \
-world_size 1 -gpu_ranks 0

CUDA_VISIBLE_DEVICES=0 \
onmt_train \
-word_vec_size 300 \
-pre_word_vecs_enc "data/$dataset/$data/user.embeddings.enc.pt" \
-pre_word_vecs_dec "data/$dataset/$data/user.embeddings.dec.pt" \
-data data/$dataset/$data/user \
-save_model nmt/$dataset/$data/user-glove-model \
-world_size 1 -gpu_ranks 0