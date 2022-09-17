#!/bin/bash

set -e

project_dir="fill the project path here"

checkpoint="the best checkpoint selected according to the validation set"
zero_shot_translation_dir=${project_dir}/data/opus-100-corpus/many-many/translation_zero_shot/${checkpoint}

for lang_pair in `ls ${zero_shot_translation_dir}`; do
    array=(${lang_pair//-/ })
    src_lang=${array[0]}
    tgt_lang=${array[1]}
    trans_file_path=${zero_shot_translation_dir}/${lang_pair}/sys.test.${src_lang}-${tgt_lang}.${tgt_lang}
    nohup python -u ${project_dir}/nmt/evaluation/lang_acc.py \
        --inputs ${trans_file_path} \
        --lang_code ${tgt_lang} > langacc.${lang_pair} 2>& 1&
done
