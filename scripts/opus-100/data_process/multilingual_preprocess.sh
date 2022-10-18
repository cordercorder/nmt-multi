#!/bin/bash

set -e

project_dir="fill the project path here"

# filename format in multilingual_corpus_dir
# train.${src_lang}-${tgt_lang}.${src_lang}, train.${src_lang}-${tgt_lang}.${tgt_lang}
# valid.${src_lang}-${tgt_lang}.${src_lang}, valid.${src_lang}-${tgt_lang}.${tgt_lang}
# test.${src_lang}-${tgt_lang}.${src_lang}, test.${src_lang}-${tgt_lang}.${tgt_lang}
multilingual_corpus_dir="raw corpus directory path"

root_data_dir=${project_dir}/data/opus-100-corpus/preprocessed_data

spm_data_dir=${root_data_dir}/spm_data
spm_corpus_dir=${root_data_dir}/spm_corpus
data_bin_mul_dir=${root_data_dir}/data_bin_mul
main_data_bin_dir=${root_data_dir}/main_data_bin
extra_data_bin_dir=${root_data_dir}/extra_data_bin

# vocabulary sizes
vocab_size=64000

mkdir -p ${root_data_dir} ${spm_data_dir} ${spm_corpus_dir} ${data_bin_mul_dir} ${main_data_bin_dir} ${extra_data_bin_dir}

spm_inputs=""
for lang_pair in `ls ${multilingual_corpus_dir}`; do
    array=(${lang_pair//-/ })
    src_lang=${array[0]}
    tgt_lang=${array[1]}
    parallel_corpus_dir=${multilingual_corpus_dir}/${lang_pair}

    # for English centric dataset, only use parallel corpora of xx -> en as the training data for sentencepiece
    if [[ ${tgt_lang} = "en" ]]; then
        spm_inputs+="${parallel_corpus_dir}/train.${src_lang}-${tgt_lang}.${src_lang},"
        spm_inputs+="${parallel_corpus_dir}/train.${src_lang}-${tgt_lang}.${tgt_lang},"
    fi

done

spm_inputs_len=${#spm_inputs}
spm_inputs=${spm_inputs:0:spm_inputs_len-1}

spm_train --normalization_rule_name identity --input ${spm_inputs} --model_prefix ${spm_data_dir}/spm --vocab_size ${vocab_size} --character_coverage 1.0 --model_type bpe

echo "spm training end!"

lang_sets=""

for lang_pair in `ls ${multilingual_corpus_dir}`; do
    array=(${lang_pair//-/ })
    src_lang=${array[0]}
    tgt_lang=${array[1]}
    parallel_corpus_dir=${multilingual_corpus_dir}/${lang_pair}

    output_spm_parallel_corpus_dir=${spm_corpus_dir}/${lang_pair}
    mkdir -p ${output_spm_parallel_corpus_dir}

    spm_encode --model ${spm_data_dir}/spm.model --output_format piece < ${parallel_corpus_dir}/train.${src_lang}-${tgt_lang}.${src_lang} > ${output_spm_parallel_corpus_dir}/train.${src_lang}-${tgt_lang}.${src_lang}
    spm_encode --model ${spm_data_dir}/spm.model --output_format piece < ${parallel_corpus_dir}/train.${src_lang}-${tgt_lang}.${tgt_lang} > ${output_spm_parallel_corpus_dir}/train.${src_lang}-${tgt_lang}.${tgt_lang}

    python -u ${project_dir}/nmt/data_handling/corpus_manager.py \
        --src_path ${output_spm_parallel_corpus_dir}/train.${src_lang}-${tgt_lang}.${src_lang} \
        --tgt_path ${output_spm_parallel_corpus_dir}/train.${src_lang}-${tgt_lang}.${tgt_lang} \
        --output_src_path ${output_spm_parallel_corpus_dir}/train.remove_long_sentence.${src_lang}-${tgt_lang}.${src_lang} \
        --output_tgt_path ${output_spm_parallel_corpus_dir}/train.remove_long_sentence.${src_lang}-${tgt_lang}.${tgt_lang} \
        --operation remove_long_sentence \
        --max_sentence_length 100
    
    echo "======== remove_long_sentence of training data end! ========"

    lang_sets+="${src_lang} "
    lang_sets+="${tgt_lang} "

    for corpus_type in "valid" "test"; do
        # split valid and test set into subwords
        if [[ -f ${parallel_corpus_dir}/${corpus_type}.${src_lang}-${tgt_lang}.${src_lang} ]] && [[ -f ${parallel_corpus_dir}/${corpus_type}.${src_lang}-${tgt_lang}.${tgt_lang} ]]; then
            spm_encode --model ${spm_data_dir}/spm.model --output_format piece < ${parallel_corpus_dir}/${corpus_type}.${src_lang}-${tgt_lang}.${src_lang} > ${output_spm_parallel_corpus_dir}/${corpus_type}.${src_lang}-${tgt_lang}.${src_lang}
            spm_encode --model ${spm_data_dir}/spm.model --output_format piece < ${parallel_corpus_dir}/${corpus_type}.${src_lang}-${tgt_lang}.${tgt_lang} > ${output_spm_parallel_corpus_dir}/${corpus_type}.${src_lang}-${tgt_lang}.${tgt_lang}

            # remove long sentences in valid set
            if [[ ${corpus_type} = "valid" ]]; then
                python -u ${project_dir}/nmt/data_handling/corpus_manager.py \
                    --src_path ${output_spm_parallel_corpus_dir}/${corpus_type}.${src_lang}-${tgt_lang}.${src_lang} \
                    --tgt_path ${output_spm_parallel_corpus_dir}/${corpus_type}.${src_lang}-${tgt_lang}.${tgt_lang} \
                    --output_src_path ${output_spm_parallel_corpus_dir}/${corpus_type}.remove_long_sentence.${src_lang}-${tgt_lang}.${src_lang} \
                    --output_tgt_path ${output_spm_parallel_corpus_dir}/${corpus_type}.remove_long_sentence.${src_lang}-${tgt_lang}.${tgt_lang} \
                    --operation remove_long_sentence \
                    --max_sentence_length 100
            fi
        fi
    done
done


dict_files=""

for lang_pair in `ls ${spm_corpus_dir}`; do
    spm_parallel_corpus=${spm_corpus_dir}/${lang_pair}
    array=(${lang_pair//-/ })
    src_lang=${array[0]}
    tgt_lang=${array[1]}
    destdir=${data_bin_mul_dir}/${src_lang}-${tgt_lang}
    mkdir -p ${destdir}
    fairseq-preprocess \
        --source-lang ${src_lang} \
        --target-lang ${tgt_lang} \
        --trainpref ${spm_parallel_corpus}/train.remove_long_sentence.${src_lang}-${tgt_lang} \
        --destdir ${destdir} \
        --workers 32
    echo ${lang_pair} end!
    dict_files+="${destdir}/dict.${src_lang}.txt "
    dict_files+="${destdir}/dict.${tgt_lang}.txt "
done

python -u ${project_dir}/nmt/data_handling/merge_dict.py \
    --dict_files ${dict_files} \
    --merged_dict ${main_data_bin_dir}/dict.txt \
    --finalize

echo "merge dict end!"

lang_pairs=""
extra_lang_pairs=""

for lang_pair in `ls ${spm_corpus_dir}`; do
    spm_parallel_corpus=${spm_corpus_dir}/${lang_pair}
    array=(${lang_pair//-/ })
    src_lang=${array[0]}
    tgt_lang=${array[1]}

    options=""
    
    if [[ -f ${spm_parallel_corpus}/valid.remove_long_sentence.${src_lang}-${tgt_lang}.${src_lang} ]] && [[ -f ${spm_parallel_corpus}/valid.remove_long_sentence.${src_lang}-${tgt_lang}.${tgt_lang} ]]; then
        options+="--validpref ${spm_parallel_corpus}/valid.remove_long_sentence.${src_lang}-${tgt_lang} "
        lang_pairs+="${lang_pair}\n"
    else
        extra_lang_pairs+="${lang_pair}\n"
    fi

    if [[ -f ${spm_parallel_corpus}/test.remove_long_sentence.${src_lang}-${tgt_lang}.${src_lang} ]] && [[ -f ${spm_parallel_corpus}/test.remove_long_sentence.${src_lang}-${tgt_lang}.${tgt_lang} ]]; then
        options+="--testpref ${spm_parallel_corpus}/test.remove_long_sentence.${src_lang}-${tgt_lang} "
    fi

    if [[ -f ${spm_parallel_corpus}/test.${src_lang}-${tgt_lang}.${src_lang} ]] && [[ -f ${spm_parallel_corpus}/test.${src_lang}-${tgt_lang}.${tgt_lang} ]]; then
        options+="--testpref ${spm_parallel_corpus}/test.${src_lang}-${tgt_lang} "
    fi

    destdir=${main_data_bin_dir}

    if [[ ${#options} -eq 0 ]]; then
        destdir=${extra_data_bin_dir}
    fi

    fairseq-preprocess \
        --source-lang ${src_lang} \
        --target-lang ${tgt_lang} \
        --srcdict ${main_data_bin_dir}/dict.txt \
        --tgtdict ${main_data_bin_dir}/dict.txt \
        --trainpref ${spm_parallel_corpus}/train.remove_long_sentence.${src_lang}-${tgt_lang} \
        --destdir ${destdir} \
        --workers 32 \
        ${options}

    echo "${lang_pair} end!"
done

# remove duplicated langs
lang_sets=$(python -c "print(' '.join(set('${lang_sets}'.split())))")
lang_sets=${lang_sets// /\\n}

echo -e ${lang_sets} > ${root_data_dir}/lang_dict.txt
echo -e ${lang_pairs} > ${root_data_dir}/lang_pairs.txt
echo -e ${extra_lang_pairs} > ${root_data_dir}/extra_lang_pairs.txt
