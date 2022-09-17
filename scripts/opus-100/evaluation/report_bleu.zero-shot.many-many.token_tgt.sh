#!/bin/bash

set -e

project_dir="fill the project path here"

raw_zero_shot_corpus_dir="fill the raw corpus path here"

zero_shot_translation_output_dir=${project_dir}/data/opus-100-corpus/many-many.token_tgt/translation_zero_shot

lang_pairs="ar-de,ar-fr,ar-nl,ar-ru,ar-zh,de-fr,de-nl,de-ru,de-zh,fr-nl,fr-ru,fr-zh,nl-ru,nl-zh,ru-zh,de-ar,fr-ar,nl-ar,ru-ar,zh-ar,fr-de,nl-de,ru-de,zh-de,nl-fr,ru-fr,zh-fr,ru-nl,zh-nl,zh-ru"

checkpoint="the best checkpoint selected according to the validation set on supervised translation directions"

zero_thot_translation_dir=${zero_shot_translation_output_dir}/${checkpoint}
for lang_pair in ${lang_pairs//,/ }; do
    array=(${lang_pair//-/ })
    src_lang=${array[0]}
    tgt_lang=${array[1]}
    parallel_trans_dir=${zero_thot_translation_dir}/${lang_pair}

    score=$(sacrebleu -w 6 ${raw_zero_shot_corpus_dir}/${lang_pair}/opus.${src_lang}-${tgt_lang}-test.${tgt_lang} < ${parallel_trans_dir}/sys.test.${src_lang}-${tgt_lang}.${tgt_lang})
    echo ${checkpoint}.${lang_pair}.${score}
done
