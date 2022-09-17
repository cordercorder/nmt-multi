#!/bin/bash

set -e

project_dir="fill the project path here"

lang_pairs="ar,bg,el,sr,eu,ku,ja,fi,da,sq,cs,hy,ur,pt_br,gl,es,az,be,hu,fa,mk,uk,vi,bs,sv,de,ta,ms,nl,eo,ko,ru,et,mn,id,he,tr,sk,sl,zh_cn,bn,mr,hr,fr,nb,lt,th,fr_ca,zh_tw,ka,pl,hi,ro,pt,zh,kk,it,my"

spm_corpus_dir=${project_dir}/data/ted/preprocessed_data/spm_corpus
output_corpus_dir=${project_dir}/data/ted/preprocessed_data/spm_corpus_extract_parallel_data

for src_lang in ${lang_pairs//,/ }; do
    for tgt_lang in ${lang_pairs//,/ }; do
        if [[ ${src_lang} = ${tgt_lang} ]]; then
            continue
        fi

        if [[ -d ${spm_corpus_dir}/${src_lang}-en ]]; then
            lang_pair_a=${src_lang}-en
        elif [[ -d ${spm_corpus_dir}/en-${src_lang} ]]; then
            lang_pair_a=en-${src_lang}
        fi

        if [[ -d ${spm_corpus_dir}/${tgt_lang}-en ]]; then
            lang_pair_b=${tgt_lang}-en
        elif [[ -d ${spm_corpus_dir}/en-${tgt_lang} ]]; then
            lang_pair_b=en-${tgt_lang}
        else
            exit 1
        fi

        output_parallel_corpus_dir=${output_corpus_dir}/${src_lang}-${tgt_lang}
        mkdir -p ${output_parallel_corpus_dir}

        python -u ${project_dir}/nmt/data_handling/extract_parallel_data_from_centric_corpus.py \
            --src_corpus_a ${spm_corpus_dir}/${lang_pair_a}/test.${lang_pair_a}.en \
            --tgt_corpus_a ${spm_corpus_dir}/${lang_pair_a}/test.${lang_pair_a}.${src_lang} \
            --src_corpus_b ${spm_corpus_dir}/${lang_pair_b}/test.${lang_pair_b}.en \
            --tgt_corpus_b ${spm_corpus_dir}/${lang_pair_b}/test.${lang_pair_b}.${tgt_lang} \
            --output_src_data_path ${output_parallel_corpus_dir}/test.${src_lang}-${tgt_lang}.${src_lang} \
            --output_tgt_data_path ${output_parallel_corpus_dir}/test.${src_lang}-${tgt_lang}.${tgt_lang}
        
        echo "${src_lang}-${tgt_lang} end!"
    done
done
