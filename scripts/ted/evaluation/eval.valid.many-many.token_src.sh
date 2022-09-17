#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0

project_dir="fill the project path here"

lang_dict=${project_dir}/data/ted/preprocessed_data/lang_dict.txt
data_bin_dir=${project_dir}/data/ted/preprocessed_data/main_data_bin

save_dir=${project_dir}/data/ted/many-many.token_src/checkpoint
translation_output_dir=${project_dir}/data/ted/many-many.token_src/translation_valid

max_tokens=8000
batch_size=512

lang_pairs="en-hu,en-eo,en-es,en-ka,en-nb,en-az,en-da,en-eu,en-id,en-cs,en-zh_tw,en-sv,en-sq,en-be,en-bs,en-bn,en-hi,en-fr,en-sk,en-vi,en-hy,en-ro,en-fa,en-ko,en-it,en-ur,en-pt_br,en-ja,en-zh,en-ar,en-fi,en-my,en-mn,en-ta,en-th,en-el,en-et,en-bg,en-tr,en-sl,en-de,en-mr,en-hr,en-pl,en-lt,en-gl,en-sr,en-pt,en-ku,en-uk,en-zh_cn,en-kk,en-ms,en-nl,en-he,en-ru,en-mk,en-fr_ca,hu-en,eo-en,es-en,ka-en,nb-en,az-en,da-en,eu-en,id-en,cs-en,zh_tw-en,sv-en,sq-en,be-en,bs-en,bn-en,hi-en,fr-en,sk-en,vi-en,hy-en,ro-en,fa-en,ko-en,it-en,ur-en,pt_br-en,ja-en,zh-en,ar-en,fi-en,my-en,mn-en,ta-en,th-en,el-en,et-en,bg-en,tr-en,sl-en,de-en,mr-en,hr-en,pl-en,lt-en,gl-en,sr-en,pt-en,ku-en,uk-en,zh_cn-en,kk-en,ms-en,nl-en,he-en,ru-en,mk-en,fr_ca-en"

checkpoint_array=(
    "all saved checkpoints"
)

for ((i=0;i<${#checkpoint_array[@]};i++)); do
    checkpoint=${checkpoint_array[i]}
    translation_dir=${translation_output_dir}/${checkpoint}
    mkdir -p ${translation_dir}

    python -u ${project_dir}/nmt/evaluation/generate.py ${data_bin_dir} \
        --user-dir ${project_dir}/nmt/user_dir \
        --task translation_multi_simple_epoch \
        --evaluation-lang-pairs ${lang_pairs} \
        --translation-output-dir ${translation_dir} \
        --lang-pairs ${lang_pairs} \
        --lang-dict ${lang_dict} \
        --source-dict ${data_bin_dir}/dict.txt \
        --target-dict ${data_bin_dir}/dict.txt \
        --encoder-langtok tgt \
        --gen-subset valid \
        --path ${save_dir}/${checkpoint} \
        --batch-size ${batch_size} \
        --max-tokens ${max_tokens} \
        --beam 5 \
        --post-process sentencepiece

    echo "${checkpoint} complete!"
done
