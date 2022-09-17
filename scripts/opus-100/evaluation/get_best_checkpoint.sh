#!/bin/bash

set -e

project_dir="fill the project path here"

python -u ${project_dir}/nmt/evaluation/get_best_checkpoint.py \
    --valid_bleu_statistic "bleu statistic on validation set"
