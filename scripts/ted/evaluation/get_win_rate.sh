#!/bin/bash

set -e

project_dir="fill the project path here"

python -u ${project_dir}/nmt/evaluation/get_win_rate.py \
    --baseline "baseline" \
    --others "data compared with baseline" \
    --result_path "output win rate data" \
    --pivot_lang en
