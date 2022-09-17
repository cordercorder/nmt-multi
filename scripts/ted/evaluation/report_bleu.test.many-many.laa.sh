#!/bin/bash

set -e

project_dir="fill the project path here"

root_data_dir=${project_dir}/data/ted/preprocessed_data
raw_corpus_dir=${root_data_dir}/spm_corpus

translation_output_dir=${project_dir}/data/ted/many-many.laa/translation_test

lang_pairs="en-hu,en-eo,en-es,en-ka,en-nb,en-az,en-da,en-eu,en-id,en-cs,en-zh_tw,en-sv,en-sq,en-be,en-bs,en-bn,en-hi,en-fr,en-sk,en-vi,en-hy,en-ro,en-fa,en-ko,en-it,en-ur,en-pt_br,en-ja,en-zh,en-ar,en-fi,en-my,en-mn,en-ta,en-th,en-el,en-et,en-bg,en-tr,en-sl,en-de,en-mr,en-hr,en-pl,en-lt,en-gl,en-sr,en-pt,en-ku,en-uk,en-zh_cn,en-kk,en-ms,en-nl,en-he,en-ru,en-mk,en-fr_ca,hu-en,eo-en,es-en,ka-en,nb-en,az-en,da-en,eu-en,id-en,cs-en,zh_tw-en,sv-en,sq-en,be-en,bs-en,bn-en,hi-en,fr-en,sk-en,vi-en,hy-en,ro-en,fa-en,ko-en,it-en,ur-en,pt_br-en,ja-en,zh-en,ar-en,fi-en,my-en,mn-en,ta-en,th-en,el-en,et-en,bg-en,tr-en,sl-en,de-en,mr-en,hr-en,pl-en,lt-en,gl-en,sr-en,pt-en,ku-en,uk-en,zh_cn-en,kk-en,ms-en,nl-en,he-en,ru-en,mk-en,fr_ca-en"

checkpoint="the best checkpoint selected according to the validation set"

translation_dir=${translation_output_dir}/${checkpoint}
for lang_pair in ${lang_pairs//,/ }; do
    array=(${lang_pair//-/ })
    src_lang=${array[0]}
    tgt_lang=${array[1]}
    parallel_trans_dir=${translation_dir}/${lang_pair}

    options="-l "
    if [[ ${tgt_lang:0:2} = "zh" ]]; then
        options+="${src_lang}-${tgt_lang:0:2}"
    else
        options+="${src_lang}-${tgt_lang}"
    fi
    
    sacremoses -l ${tgt_lang:0:2} detokenize < ${parallel_trans_dir}/sys.test.${src_lang}-${tgt_lang}.${tgt_lang} > ${parallel_trans_dir}/sys.test.detok.${src_lang}-${tgt_lang}.${tgt_lang}

    score=$(sacrebleu -w 6 ${options} ${spm_corpus_dir}/${lang_pair}/test.detok.${src_lang}-${tgt_lang}.${tgt_lang} < ${parallel_trans_dir}/sys.test.detok.${src_lang}-${tgt_lang}.${tgt_lang})
    echo ${checkpoint}.${lang_pair}.${score}
done

