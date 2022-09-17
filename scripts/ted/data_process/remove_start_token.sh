#!/bin/bash

set -e

project_dir="fill the project path here"

# filename format in multilingual_corpus_dir
# train.${src_lang}-${tgt_lang}.${src_lang}, train.${src_lang}-${tgt_lang}.${tgt_lang}
# valid.${src_lang}-${tgt_lang}.${src_lang}, valid.${src_lang}-${tgt_lang}.${tgt_lang}
# test.${src_lang}-${tgt_lang}.${src_lang}, test.${src_lang}-${tgt_lang}.${tgt_lang}
multilingual_corpus_dir="raw corpus directory path"
output_multilingual_corpus_dir="output corpus directory path"

mkdir -p ${output_multilingual_corpus_dir}

for lang_pair in `ls ${multilingual_corpus_dir}`; do
    array=(${lang_pair//-/ })
    src_lang=${array[0]}
    tgt_lang=${array[1]}

    mkdir -p ${output_multilingual_corpus_dir}/${lang_pair}
    for corpus_type in "train" "valid" "test"; do
        if [[ ${src_lang} = "en" ]]; then
            options="--remove_token_tgt"
        else
            options="--remove_token_src"
        fi
        python -u ${project_dir}/nmt/data_handling/remove_start_token.py \
            --src_corpus ${multilingual_corpus_dir}/${lang_pair}/${corpus_type}.${src_lang}-${tgt_lang}.${src_lang} \
            --tgt_corpus ${multilingual_corpus_dir}/${lang_pair}/${corpus_type}.${src_lang}-${tgt_lang}.${tgt_lang} \
            ${options} \
            --output_src_corpus ${output_multilingual_corpus_dir}/${lang_pair}/${corpus_type}.${src_lang}-${tgt_lang}.${src_lang} \
            --output_tgt_corpus ${output_multilingual_corpus_dir}/${lang_pair}/${corpus_type}.${src_lang}-${tgt_lang}.${tgt_lang}
        echo ${lang_pair}.${options}
    done
    
done
