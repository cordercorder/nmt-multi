#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
n_process=6
master_port=12345

project_dir="fill the project path here"

root_data_dir=${project_dir}/data/opus-100-corpus/preprocessed_data

main_data_bin_dir=${root_data_dir}/main_data_bin
extra_data_bin_dir=${root_data_dir}/extra_data_bin
lang_dict=${root_data_dir}/lang_dict.txt

save_dir=${project_dir}/data/opus-100-corpus/many-many.token_src/checkpoint
tensorboard_logdir=${project_dir}/data/opus-100-corpus/many-many.token_src/tensorboard_logdir

mkdir -p ${save_dir} ${tensorboard_logdir}

lang_pairs="en-fr,cy-en,hu-en,en-lt,en-mg,yi-en,as-en,en-mr,uz-en,eo-en,li-en,es-en,ka-en,am-en,en-he,en-ja,nb-en,en-ku,en-cs,en-fi,si-en,en-no,en-se,az-en,en-ga,da-en,en-vi,eu-en,en-pa,ca-en,id-en,en-eu,cs-en,kn-en,te-en,en-ug,en-be,rw-en,gu-en,en-cy,en-tt,en-am,xh-en,en-nb,sv-en,sq-en,en-nn,en-bn,ha-en,en-hu,en-pl,en-ko,en-tg,en-zu,en-nl,ps-en,af-en,be-en,ga-en,mg-en,en-mt,bs-en,or-en,bn-en,en-sr,tg-en,hi-en,fr-en,se-en,en-hr,en-eo,en-de,en-it,sk-en,tt-en,is-en,km-en,en-br,nn-en,vi-en,en-ka,ne-en,en-et,ro-en,en-ha,fa-en,oc-en,en-sh,ko-en,en-yi,en-fa,it-en,no-en,en-ig,en-af,en-da,en-th,ur-en,en-pt,zu-en,ja-en,zh-en,ar-en,en-ky,fi-en,en-mk,lv-en,my-en,en-kk,ta-en,en-ca,mt-en,fy-en,en-uk,th-en,el-en,ml-en,et-en,en-my,en-es,en-sv,wa-en,en-sk,en-ro,en-oc,bg-en,en-uz,tr-en,sl-en,sh-en,de-en,en-lv,en-is,en-km,mr-en,en-hi,pa-en,en-gu,hr-en,en-tk,en-ta,pl-en,en-kn,lt-en,en-ps,ug-en,en-bg,br-en,en-ru,en-sl,en-ne,en-te,en-bs,tk-en,gl-en,en-si,en-rw,sr-en,pt-en,en-tr,ky-en,en-gd,ku-en,en-id,en-ur,en-li,uk-en,en-or,en-sq,gd-en,en-ar,en-ml,kk-en,en-el,en-zh,en-gl,en-as,ig-en,ms-en,nl-en,en-fy,en-az,he-en,en-ms,ru-en,mk-en,en-wa,en-xh"

extra_lang_pairs="en-an,en-mn,en-dz,hy-en,mn-en,en-hy,dz-en,yo-en,an-en,en-yo"

python -m torch.distributed.launch --nproc_per_node=${n_process} \
    --master_addr="127.0.0.1" \
    --master_port=${master_port} \
    $(which fairseq-train) ${main_data_bin_dir} \
    --distributed-world-size ${n_process} \
    --task translation_multi_simple_epoch \
    --extra-data "{\"extra\": \"${extra_data_bin_dir}\"}" \
    --extra-lang-pairs "{\"extra\": \"${extra_lang_pairs}\"}" \
    --langtoks "{\"extra\": (\"tgt\", None)}" \
    --arch transformer \
    --layernorm-embedding \
    --sampling-method "temperature" \
    --sampling-temperature 5 \
    --encoder-langtok tgt \
    --lang-dict ${lang_dict} \
    --lang-pairs ${lang_pairs} \
    --source-dict ${main_data_bin_dir}/dict.txt \
    --target-dict ${main_data_bin_dir}/dict.txt \
    --share-all-embeddings \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.0005 \
    --stop-min-lr 1e-09 \
    --dropout 0.1 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --update-freq 4 \
    --max-epoch 30 \
    --save-interval-updates 5000 \
    --save-dir ${save_dir} \
    --keep-best-checkpoints 5 \
    --keep-interval-updates 5 \
    --skip-invalid-size-inputs-valid-test \
    --tensorboard-logdir ${tensorboard_logdir} \
    --fp16
