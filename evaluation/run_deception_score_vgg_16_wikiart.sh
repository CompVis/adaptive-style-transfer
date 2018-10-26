#!/usr/bin/env bash
set -e
LOG_DIR=logs
mkdir -p ${LOG_DIR}

NET=vgg_16_multihead

METHODS=( "ours" )
#METHODS=( "ours" "real_wiki_test" )

for method in ${METHODS[@]}
do
        echo $method
        python eval_deception_score.py \
        -net=${NET} \
        -s="evaluation_data/model.ckpt-790000" \
        -log=${LOG_DIR}/deception_score_${method}.txt \
        --method=$method \
        --num_classes=624 \
        --dataset="wikiart"
done
