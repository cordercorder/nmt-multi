#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0

project_dir="fill the project path here"

root_data_dir=${project_dir}/data/opus-100-corpus/preprocessed_data
lang_dict=${root_data_dir}/lang_dict.txt
zero_shot_data_bin_dir=${root_data_dir}/zero_shot_data_bin

save_dir=${project_dir}/data/opus-100-corpus/many-many.token_src/checkpoint
translation_output_dir=${project_dir}/data/opus-100-corpus/many-many.token_src/translation_zero_shot

max_tokens=8000
batch_size=512

lang_pairs="en-fr,cy-en,hu-en,en-lt,en-mg,yi-en,as-en,en-mr,uz-en,eo-en,li-en,es-en,ka-en,am-en,en-he,en-ja,nb-en,en-ku,en-cs,en-fi,si-en,en-no,en-se,az-en,en-ga,da-en,en-vi,eu-en,en-pa,ca-en,id-en,en-eu,cs-en,kn-en,te-en,en-ug,en-be,rw-en,gu-en,en-cy,en-tt,en-am,xh-en,en-nb,sv-en,sq-en,en-nn,en-bn,ha-en,en-hu,en-pl,en-ko,en-tg,en-zu,en-nl,ps-en,af-en,be-en,ga-en,mg-en,en-mt,bs-en,or-en,bn-en,en-sr,tg-en,hi-en,fr-en,se-en,en-hr,en-eo,en-de,en-it,sk-en,tt-en,is-en,km-en,en-br,nn-en,vi-en,en-ka,ne-en,en-et,ro-en,en-ha,fa-en,oc-en,en-sh,ko-en,en-yi,en-fa,it-en,no-en,en-ig,en-af,en-da,en-th,ur-en,en-pt,zu-en,ja-en,zh-en,ar-en,en-ky,fi-en,en-mk,lv-en,my-en,en-kk,ta-en,en-ca,mt-en,fy-en,en-uk,th-en,el-en,ml-en,et-en,en-my,en-es,en-sv,wa-en,en-sk,en-ro,en-oc,bg-en,en-uz,tr-en,sl-en,sh-en,de-en,en-lv,en-is,en-km,mr-en,en-hi,pa-en,en-gu,hr-en,en-tk,en-ta,pl-en,en-kn,lt-en,en-ps,ug-en,en-bg,br-en,en-ru,en-sl,en-ne,en-te,en-bs,tk-en,gl-en,en-si,en-rw,sr-en,pt-en,en-tr,ky-en,en-gd,ku-en,en-id,en-ur,en-li,uk-en,en-or,en-sq,gd-en,en-ar,en-ml,kk-en,en-el,en-zh,en-gl,en-as,ig-en,ms-en,nl-en,en-fy,en-az,he-en,en-ms,ru-en,mk-en,en-wa,en-xh"

zero_shot_lang_pairs="ar-de,ar-fr,ar-nl,ar-ru,ar-zh,de-fr,de-nl,de-ru,de-zh,fr-nl,fr-ru,fr-zh,nl-ru,nl-zh,ru-zh,de-ar,fr-ar,nl-ar,ru-ar,zh-ar,fr-de,nl-de,ru-de,zh-de,nl-fr,ru-fr,zh-fr,ru-nl,zh-nl,zh-ru"

checkpoint="the best checkpoint selected according to the validation set on supervised translation directions"
translation_dir=${translation_output_dir}/${checkpoint}
mkdir -p ${translation_dir}

python -u ${project_dir}/nmt/evaluation/generate.py ${zero_shot_data_bin_dir} \
    --user-dir ${project_dir}/nmt/user_dir \
    --task translation_multi_simple_epoch \
    --evaluation-lang-pairs ${zero_shot_lang_pairs} \
    --translation-output-dir ${translation_dir} \
    --lang-pairs ${lang_pairs} \
    --lang-dict ${lang_dict} \
    --source-dict ${zero_shot_data_bin_dir}/dict.txt \
    --target-dict ${zero_shot_data_bin_dir}/dict.txt \
    --encoder-langtok tgt \
    --gen-subset test \
    --path ${save_dir}/${checkpoint} \
    --max-tokens ${max_tokens} \
    --batch-size ${batch_size} \
    --beam 5 \
    --post-process sentencepiece

echo "${checkpoint} complete!"
