# Tabel of Contents

- [Requirements](#requirements)
- [Usage](#usage)
	- [Prepare Data](#prepare-data)
	- [ITA](#ita)
		- [Imaginator](#imaginator)
		- [Arbitrator](#arbitrator)
	- [Baseline](#baseline)
	- [Run Random Classifier on Test set](#run-random-classifier-on-test-set)
	- [Ablation Study: Test different Imaginators generation performance on TextCNNs based Arbitrator](#test-different-imaginators-generation-performance-on-textCNNs-based-arbitrator)
	- [Sample results from generated utterances](#sample-results-from-generated-utterances)
# Requirements

* This work was tested with PyTorch 1.3.1, CUDA 10.1, Python 3.6
* Build the required dependencies from the following command

```shell
pip install -r requirements.txt
```

# Usage

## Prepare Data

### Download pre-trained word vectors

```shell
wget -P data/ http://nlp.stanford.edu/data/glove.6B.zip && unzip -d data/ data/glove.6B.zip 
```

###  Preprocess data

```bash
# For MultiWoz Dataset
python utils/utils_MultiWoz.py
# For DailyDialogue Dataset
python utils/utils_DailyDialog.py
# For CCPE Dataset
python utils/utils_CCPE.py
```
## ITA

> MultiWoz as an example

### Imaginator

#### Prepare data for Imaginator

```shell
sh prepare_data_for_imaginator.sh
```

#### Training for Imaginator

* **LSTM for MultiWoz or DailyDialogue**

```bash
# sh train_LSTM.sh [data_name: multiwoz | dailydialog]
sh train_LSTM.sh multiwoz
```

* **LSTM+Attn. for MultiWoz or DailyDialogue**

```bash
# sh train_LSTM_attn.sh [data_name: multiwoz | dailydialog]
sh train_LSTM_attn.sh multiwoz
```

* **LSTM+Attn.+GloVe  for  MultiWoz or DailyDialogue**

```bash
# sh train_LSTM_attn_GloVe.sh [data_name: multiwoz | dailydialog]
sh train_LSTM_attn_GloVe.sh multiwoz
```

* **All three type of LSTM for CCPE**

```bash
sh train_LSTM_all_for_CCPE.sh
```

#### Inferencing for Imaginator

* **LSTM**

Note: `woattn`  means LSTM without attn.

````bash
# sh infer_LSTM.sh [dataset_name] [best_agent_ckpt_name] [best_user_ckpt_name]
# Example
# for MultiWoz 
sh infer_LSTM.sh multiwoz agent-woattn-model_step_90000.pt user-woattn-model_step_90000.pt
````

* **LSTM + Attn.**

Note: `origin` means LSTM with attn.

```bash
# sh infer_LSTM_attn.sh [dataset_name] [best_agent_ckpt_name] [best_user_ckpt_name]
# Example
# for MultiWoz 
sh infer_LSTM_attn.sh multiwoz agent-origin-model_step_70000.pt user-origin-model_step_80000.pt
```

* **LSTM + Attn. + GloVe**

Note: `glove` means LSTM with attn. and GloVe pretrained word vectors

```shell
# sh infer_LSTM_attn_GloVe.sh [dataset_name] [best_agent_ckpt_name] [best_user_ckpt_name]
# Example
# for MultiWoz 
sh infer_LSTM_attn_GloVe.sh multiwoz agent-glove-model_step_80000.pt user-glove-model_step_80000.pt
```

#### Evaluate for Imaginator

```shell
# for MultiWoz 
sh eval_BLEU.sh multiwoz
```

### Arbitrator

#### Prepare data for Arbitrator

```shell
# sh prepare_data_for_arbitrator.sh [data_name: multiwoz | dailydialog | ccpe]
# for MultiWoz
sh prepare_data_for_arbitrator.sh multiwoz
```

#### Training, Validation, Testing for Arbitrator

* **TextCNN-ITA**

```shell
#CUDA_VISIBLE_DEVICES=0 python trainTextCNN-ITA.py \
#--data_name [data_name] \
#--nmt_name [origin, glove, woattn, default: origin] \
#--do_eval --do_train \
#-bsz [batch size] -kn [kernel num] \
#-dr [dr] -ksz [kernel sizes]
# Example:
CUDA_VISIBLE_DEVICES=0 python trainTextCNN-ITA.py --data_name multiwoz --do_eval --do_train -bsz=64 -kn=400 -dr=0.3 -ksz 7 8 9
```

```shell
# run TextCNN-ITA or TextCNN from checkpoint
# Example:
CUDA_VISIBLE_DEVICES=0 python trainTextCNN-ITA.py --data_name multiwoz --do_eval --ckpt checkpoints/ckpt.pt
```

* **GRU-ITA**

```shell
#CUDA_VISIBLE_DEVICES=0 python trainGRU-ITA.py \
#--data_name [data_name] \
#--nmt_name [origin, glove, woattn, default: origin] \
#--do_eval --do_train \
#-bsz [batch size] -hdim [hidden size] \
#-dr [dr]
# Example:
CUDA_VISIBLE_DEVICES=0 python trainGRU-ITA.py --data_name multiwoz --do_eval --do_train -bsz 32 -dr 0.3 -hdim 300
```

```shell
# run GRU or GRU-ITA from checkpoint
# Example:
CUDA_VISIBLE_DEVICES=0 python trainGRU-ITA.py --data_name multiwoz --do_eval --ckpt checkpoints/ckpt.pt
```

* **BERT-ITA**

```shell
# sh train_BERT_ITA.sh [data_name: multiwoz | dailydialog | ccpe]
# for MultiWoz
sh train_BERT_ITA.sh multiwoz
```

# Baseline

* **Baseline: TextCNN**

```shell
#CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py \
#--data_name [data_name] \
#--nmt_name [origin, glove, woattn, default: origin] \
#--do_eval --do_train \
#--single_mode \ using the baseline parameters
# Example:
# for MultiWoz
CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py --data_name multiwoz --do_eval --do_train --single_mode
```


* **Baseline: GRU**

```shell
#CUDA_VISIBLE_DEVICES=0 python trainGRU.py \
#--data_name [data_name] \
#--nmt_name [origin, glove, woattn, default: origin] \
#--do_eval --do_train \
#--single_mode \ using the baseline parameters
# Example:
# for MultiWoz
CUDA_VISIBLE_DEVICES=0 python trainGRU.py --data_name multiwoz --do_eval --do_train --single_mode
```


* **Baseline: BERT**

```shell
# sh train_BERT.sh [data_name: multiwoz | dailydialog | ccpe]
# for MultiWoz
sh train_BERT.sh multiwoz
```

# Run Random Classifier on Test set

```shell
# python random_classifier.py --data_name [data_name]
# Example:
python random_classifier.py --data_name multiwoz
```

# Test different Imaginators generation performance on TextCNNs based Arbitrator

```shell
# data_name: multiwoz | dailydialog | ccpe
python trainTextCNN-ITA.py --do_eval --data_name [data_name] --ckpt [best_ckpt_path] --nmt_name woattn # woattn means LSTM without attention
python trainTextCNN-ITA.py --do_eval --data_name [data_name] --ckpt [best_ckpt_path] --nmt_name origin # origin means LSTM with attention
python trainTextCNN-ITA.py --do_eval --data_name [data_name] --ckpt [best_ckpt_path] --nmt_name glove # glove means LSTM+attention+GloVe
```

# Sample results from generated utterances

* Example: MultiWoz.  (The commands for other datasets are the same as MultiWoz)

```shell
# for MultiWoz
# baseline
CUDA_VISIBLE_DEVICES=0 python trainTextCNN.py --data_name multiwoz --do_sample --ckpt [ckpt_basline_path] --single_mode
# ours
CUDA_VISIBLE_DEVICES=0 python trainTextCNN-ITA.py --data_name multiwoz --do_sample --ckpt 
[ckpt_basline_path]
# do sampling and the output is in 'output' folder.
python sample.py --data_name multiwoz
```
