#!/bin/bash

set -e

project_dir="fill the project path here"

python -u ${project_dir}/nmt/evaluation/multilingual_bleu_statistics.py \
    --input "logs of report_bleu" \
    --output_json_data "output json data"
