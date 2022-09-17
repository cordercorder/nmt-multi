#!/bin/bash

set -e

project_dir="fill the project path here"

python -u ${project_dir}/nmt/evaluation/get_win_rate.py \
    --baseline "baseline" \
    --others "the system to be compared" \
    --result_path "output win rate data" \
    --pivot_lang en
