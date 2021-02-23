#!/bin/bash

set -e

# for DailyDialog
onmt_preprocess \
-train_src data/newDailyDialog/DailyUser/src-train-agent.txt \
-train_tgt data/newDailyDialog/DailyUser/tgt-train-agent.txt \
-valid_src data/newDailyDialog/DailyUser/src-val-agent.txt \
-valid_tgt data/newDailyDialog/DailyUser/tgt-val-agent.txt \
-save_data data/newDailyDialog/DailyUser/agent
onmt_preprocess \
-train_src data/newDailyDialog/DailyUser/src-train-user.txt \
-train_tgt data/newDailyDialog/DailyUser/tgt-train-user.txt \
-valid_src data/newDailyDialog/DailyUser/src-val-user.txt \
-valid_tgt data/newDailyDialog/DailyUser/tgt-val-user.txt \
-save_data data/newDailyDialog/DailyUser/user
# for MultiWoz
onmt_preprocess \
-train_src data/newAgentUser/newSpUser/src-train-agent.txt \
-train_tgt data/newAgentUser/newSpUser/tgt-train-agent.txt \
-valid_src data/newAgentUser/newSpUser/src-val-agent.txt \
-valid_tgt data/newAgentUser/newSpUser/tgt-val-agent.txt \
-save_data data/newAgentUser/newSpUser/agent
onmt_preprocess \
-train_src data/newAgentUser/newSpUser/src-train-user.txt \
-train_tgt data/newAgentUser/newSpUser/tgt-train-user.txt \
-valid_src data/newAgentUser/newSpUser/src-val-user.txt \
-valid_tgt data/newAgentUser/newSpUser/tgt-val-user.txt \
-save_data data/newAgentUser/newSpUser/user
# for CCPE
onmt_preprocess \
-train_src data/newCCPE/CCPEAgent/src-train-agent.txt \
-train_tgt data/newCCPE/CCPEAgent/tgt-train-agent.txt \
-valid_src data/newCCPE/CCPEAgent/src-val-agent.txt \
-valid_tgt data/newCCPE/CCPEAgent/tgt-val-agent.txt \
-save_data data/newCCPE/CCPEAgent/agent
onmt_preprocess \
-train_src data/newCCPE/CCPEAgent/src-train-user.txt \
-train_tgt data/newCCPE/CCPEAgent/tgt-train-user.txt \
-valid_src data/newCCPE/CCPEAgent/src-val-user.txt \
-valid_tgt data/newCCPE/CCPEAgent/tgt-val-user.txt \
-save_data data/newCCPE/CCPEAgent/user
