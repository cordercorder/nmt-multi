#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

project_dir="fill the project path here"

n_process=4
master_port=12346

save_dir=${project_dir}/data/ted/many-many.token_src/checkpoint

mkdir -p ${save_dir}

data_bin_dir=${project_dir}/data/ted/preprocessed_data/main_data_bin

lang_pairs="en-hu,en-eo,en-es,en-ka,en-nb,en-az,en-da,en-eu,en-id,en-cs,en-zh_tw,en-sv,en-sq,en-be,en-bs,en-bn,en-hi,en-fr,en-sk,en-vi,en-hy,en-ro,en-fa,en-ko,en-it,en-ur,en-pt_br,en-ja,en-zh,en-ar,en-fi,en-my,en-mn,en-ta,en-th,en-el,en-et,en-bg,en-tr,en-sl,en-de,en-mr,en-hr,en-pl,en-lt,en-gl,en-sr,en-pt,en-ku,en-uk,en-zh_cn,en-kk,en-ms,en-nl,en-he,en-ru,en-mk,en-fr_ca,hu-en,eo-en,es-en,ka-en,nb-en,az-en,da-en,eu-en,id-en,cs-en,zh_tw-en,sv-en,sq-en,be-en,bs-en,bn-en,hi-en,fr-en,sk-en,vi-en,hy-en,ro-en,fa-en,ko-en,it-en,ur-en,pt_br-en,ja-en,zh-en,ar-en,fi-en,my-en,mn-en,ta-en,th-en,el-en,et-en,bg-en,tr-en,sl-en,de-en,mr-en,hr-en,pl-en,lt-en,gl-en,sr-en,pt-en,ku-en,uk-en,zh_cn-en,kk-en,ms-en,nl-en,he-en,ru-en,mk-en,fr_ca-en"

lang_dict=${project_dir}/data/ted/preprocessed_data/lang_dict.txt

python -m torch.distributed.launch --nproc_per_node=${n_process} \
    --nnodes=1 --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=${master_port} \
    $(which fairseq-train) ${data_bin_dir} \
    --task translation_multi_simple_epoch \
    --arch transformer \
    --layernorm-embedding \
    --sampling-method "temperature" \
    --sampling-temperature 5 \
    --encoder-langtok tgt \
    --lang-dict ${lang_dict} \
    --lang-pairs ${lang_pairs} \
    --source-dict ${data_bin_dir}/dict.txt \
    --target-dict ${data_bin_dir}/dict.txt \
    --share-all-embeddings \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.0005 \
    --stop-min-lr 1e-09 \
    --dropout 0.2 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --update-freq 4 \
    --max-epoch 30 \
    --save-dir ${save_dir} \
    --skip-invalid-size-inputs-valid-test \
    --fp16
