#!/bin/bash

set -e


# for woattn: LSTM
CUDA_VISIBLE_DEVICES=0 onmt_train -data data/newCCPE/CCPEAgent/agent  -save_model nmt/newCCPE/CCPEAgent/agent-woattn-model -world_size 1 -gpu_ranks 0 --global_attention none -save_checkpoint_steps 200 -train_steps 4000 --valid_steps 100
CUDA_VISIBLE_DEVICES=0 onmt_train -data data/newCCPE/CCPEAgent/user -save_model nmt/newCCPE/CCPEAgent/user-woattn-model -world_size 1 -gpu_ranks 0 --global_attention none -save_checkpoint_steps 200 -train_steps 4000 --valid_steps 100

# for origin: LSTM + Global Attention
CUDA_VISIBLE_DEVICES=0 onmt_train -data data/newCCPE/CCPEAgent/agent -save_model nmt/newCCPE/CCPEAgent/agent-origin-model -world_size 1 -gpu_ranks 0 -save_checkpoint_steps 200 -train_steps 4000 --valid_steps 100
CUDA_VISIBLE_DEVICES=0 onmt_train -data data/newCCPE/CCPEAgent/user -save_model nmt/newCCPE/CCPEAgent/user-origin-model -world_size 1 -gpu_ranks 0 -save_checkpoint_steps 200 -train_steps 4000 --valid_steps 100

# for glove+attn: LSTM
python tools/embeddings_to_torch.py -emb_file_both "data/glove.6B.300d.txt" \
-dict_file "data/newCCPE/CCPEAgent/agent.vocab.pt" \
-output_file "data/newCCPE/CCPEAgent/agent.embeddings"

python tools/embeddings_to_torch.py -emb_file_both "data/glove.6B.300d.txt" \
-dict_file "data/newCCPE/CCPEAgent/user.vocab.pt" \
-output_file "data/newCCPE/CCPEAgent/user.embeddings"

CUDA_VISIBLE_DEVICES=0 \
onmt_train \
-word_vec_size 300 \
-pre_word_vecs_enc "data/newCCPE/CCPEAgent/agent.embeddings.enc.pt" \
-pre_word_vecs_dec "data/newCCPE/CCPEAgent/agent.embeddings.dec.pt" \
-data data/newCCPE/CCPEAgent/agent \
-save_model nmt/newCCPE/CCPEAgent/agent-glove-model \
-world_size 1 -gpu_ranks 0 -save_checkpoint_steps 200 -train_steps 4000 --valid_steps 100

CUDA_VISIBLE_DEVICES=0 \
onmt_train \
-word_vec_size 300 \
-pre_word_vecs_enc "data/newCCPE/CCPEAgent/user.embeddings.enc.pt" \
-pre_word_vecs_dec "data/newCCPE/CCPEAgent/user.embeddings.dec.pt" \
-data data/newCCPE/CCPEAgent/user \
-save_model nmt/newCCPE/CCPEAgent/user-glove-model \
-world_size 1 -gpu_ranks 0 -save_checkpoint_steps 200 -train_steps 4000 --valid_steps 100