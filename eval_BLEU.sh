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


for split in 'train' 'val' 'test'
do
    echo "$split"
    echo "origin {agent,user} gen model for agent his data"
    perl tools/multi-bleu.perl data/$dataset/$data/tgt-$split-agent.txt < data/$dataset/$data/pred-$split-agent-for-agent-origin.txt
    # tgt-train-agent表示agent说的话
    # agent-for-agent表示用agent的model去生成应该是agent说的话
    
    perl tools/multi-bleu.perl data/$dataset/$data/tgt-$split-agent.txt < data/$dataset/$data/pred-$split-user-for-agent-origin.txt
    # agent-for-agent表示用agent的model去生成应该是agent说的话
    

    echo "glove {agent,user} gen model for agent his data"
    perl tools/multi-bleu.perl data/$dataset/$data/tgt-$split-agent.txt < data/$dataset/$data/pred-$split-agent-for-agent-glove.txt
    # agent-for-agent表示用agent的model去生成应该是agent说的话
    


    perl tools/multi-bleu.perl data/$dataset/$data/tgt-$split-agent.txt < data/$dataset/$data/pred-$split-user-for-agent-glove.txt
    # agent-for-agent表示用agent的model去生成应该是agent说的话
    

    echo "woattn {agent,user} gen model for agent his data"
    perl tools/multi-bleu.perl data/$dataset/$data/tgt-$split-agent.txt < data/$dataset/$data/pred-$split-agent-for-agent-woattn.txt
    # agent-for-agent表示用agent的model去生成应该是agent说的话
    


    perl tools/multi-bleu.perl data/$dataset/$data/tgt-$split-agent.txt < data/$dataset/$data/pred-$split-user-for-agent-woattn.txt
    # agent-for-agent表示用agent的model去生成应该是agent说的话
    

    echo "################################"

    echo "origin {agent,user} gen model for user his data"
    perl tools/multi-bleu.perl data/$dataset/$data/tgt-$split-user.txt < data/$dataset/$data/pred-$split-agent-for-user-origin.txt
    # agent-for-agent表示用agent的model去生成应该是agent说的话
    


    perl tools/multi-bleu.perl data/$dataset/$data/tgt-$split-user.txt < data/$dataset/$data/pred-$split-user-for-user-origin.txt
    # agent-for-agent表示用agent的model去生成应该是agent说的话
    

    echo "glove {agent,user} gen model for user his data"
    perl tools/multi-bleu.perl data/$dataset/$data/tgt-$split-user.txt < data/$dataset/$data/pred-$split-agent-for-user-glove.txt
    # agent-for-agent表示用agent的model去生成应该是agent说的话
    


    perl tools/multi-bleu.perl data/$dataset/$data/tgt-$split-user.txt < data/$dataset/$data/pred-$split-user-for-user-glove.txt
    # agent-for-agent表示用agent的model去生成应该是agent说的话
    

    echo "woattn {agent,user} gen model for user his data"
    perl tools/multi-bleu.perl data/$dataset/$data/tgt-$split-user.txt < data/$dataset/$data/pred-$split-agent-for-user-woattn.txt
    # agent-for-agent表示用agent的model去生成应该是agent说的话
    


    perl tools/multi-bleu.perl data/$dataset/$data/tgt-$split-user.txt < data/$dataset/$data/pred-$split-user-for-user-woattn.txt
    # agent-for-agent表示用agent的model去生成应该是agent说的话   
    echo "################################"
done