#!/bin/bash

set -e

# prepare data for classifier

dataset=$1

if [ $dataset = 'multiwoz' ]
then
    # for MultiWoz
    python utils/utils_MultiWoz.py --combine --process_bert --nmt_name origin --addition_name directClassify
    python utils/utils_MultiWoz.py --combine --process_bert --nmt_name glove
    python utils/utils_MultiWoz.py --combine --process_bert --nmt_name woattn
elif [ $dataset = 'dailydialog' ]
then
    # for DailyDialog
    python utils/utils_DailyDialog.py --combine --process_bert --nmt_name origin --addition_name directClassify
    python utils/utils_DailyDialog.py --combine --process_bert --nmt_name woattn
    python utils/utils_DailyDialog.py --combine --process_bert --nmt_name glove
elif [ $dataset = 'ccpe' ]
then
    # for CCPE
    python utils/utils_CCPE.py --combine --process_bert --nmt_name origin --addition_name directClassify
    python utils/utils_CCPE.py --combine --process_bert --nmt_name woattn
    python utils/utils_CCPE.py --combine --process_bert --nmt_name glove
else
    echo "wrong input."
    exit 1
fi





