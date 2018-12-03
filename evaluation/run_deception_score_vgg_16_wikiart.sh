#!/usr/bin/env bash

# Copyright (C) 2018  Artsiom Sanakoyeu and Dmytro Kotovenko
#
# This file is part of Adaptive Style Transfer
#
# Adaptive Style Transfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Adaptive Style Transfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
