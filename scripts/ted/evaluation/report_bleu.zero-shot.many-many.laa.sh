#!/bin/bash

set -e

project_dir="fill the project path here"
raw_zero_shot_corpus_dir=${project_dir}/data/ted/preprocessed_data/extract_parallel_data
zero_shot_translation_output_dir=${project_dir}/data/ted/many-many.laa/translation_zero_shot

lang_pairs="az-pt,zh_tw-tr,zh-hu,hy-vi,zh_tw-uk,ta-mn,mk-eu,ru-et,cs-et,lt-gl,ru-id,it-vi,nl-ar,mn-bg,pt_br-zh_tw,ar-fa,my-th,vi-kk,eo-tr,fr_ca-hu,th-nb,fa-sk,eu-mn,fr-ku,pl-sv,cs-ur,be-tr,sr-az,ar-ru,ru-az,hy-ms,sv-gl,hy-it,zh-ku,sq-eo,pl-ja,ka-zh_tw,ro-my,uk-hi,pl-ku,uk-zh_tw,mr-eo,vi-es,el-ta,da-bg,zh_tw-ko,it-fi,ms-my,es-he,hi-ko,mr-mk,uk-ja,bn-hu,sr-id,fr_ca-bs,eu-id,ku-hu,id-bs,ka-ja,ar-bn,eo-pt_br,sk-el,da-he,hy-ka,ar-ta,pl-sk,fr-nb,ur-zh_tw,th-pl,zh_tw-bs,fr_ca-et,pt-fi,ar-he,fr_ca-sv,uk-ms,th-it,fr_ca-mr,sl-bn,ta-el,ko-es,fi-ka,ar-ro,el-bn,zh-ur,el-es,id-zh,my-kk,tr-hr,az-vi,tr-zh_cn,hy-ur,he-ms,ur-fa,eu-nl,sq-gl,ro-nl,kk-ka,be-ro,ru-el,ro-fi,th-ko,pt-mn,da-ta,fi-ko,sq-mr,pl-it,fr-pt,fr_ca-da,hr-hy,bg-sk,et-id,az-be,zh_tw-et,da-lt,ar-mk,sq-ro,he-fi,hr-sk,ro-eo,ms-ko,sl-mr,es-ur,lt-mk,vi-zh,sr-zh_cn,ur-fr,vi-lt,fr-bn,az-el,ku-ru,eu-sv,bg-eu,pt-vi,hr-fr_ca,pt-hi,eu-kk,bg-ta,sl-ku,fi-ms,it-az,az-ms,da-zh_cn,bn-nl,es-lt,eo-ur,fa-lt,sr-fi,vi-fr,it-ar,fi-de,kk-ur,sr-eu,bn-eo,ar-ko,ka-ms,sv-pt_br,bs-mr,eu-pl,eu-da,hy-gl,ku-el,sr-cs,mr-sq,nb-fi,mk-ur,sl-ko,sq-bg,lt-ro,fi-mr,it-mn,th-uk,pt_br-de,nb-sr,be-fa,he-ja,ko-eu,eu-hr,eu-lt,sk-sq,sq-sv,zh_cn-sk,bn-ar,lt-bn,ko-ka,es-fr,pt-sq,tr-hu,mk-et,gl-hu,ar-hi,fr_ca-ku,th-ro,sl-sr,pt_br-kk,fr_ca-nl,ku-ur,tr-pt_br,ar-nb,sv-uk,gl-ru,gl-fr_ca,hr-ku,fi-es,hu-fr_ca,hi-ur,zh_cn-sq,nb-da,nl-kk,ur-lt,vi-mk,el-hu,gl-bn,fi-th,be-my,zh_cn-hi,eo-de,ms-mr,eu-ur,ur-zh,bg-az,el-gl,zh_cn-ro,da-tr,bg-hu,az-eo,ku-zh_cn,it-sr,zh_tw-hu,sk-fr_ca,ar-fr_ca,it-sq,sq-hu,sr-uk,gl-ar,ro-sl,ur-ku,el-ja,ku-fr,pt_br-sv,hr-fa,ku-sk,fi-mn,sk-gl,hu-de,bg-my,hr-sv,fr-pl,nl-el,pt_br-ja,eu-ta,el-it,sv-hy,zh-mk,hu-ta,ko-lt,he-ro,fr_ca-ar,lt-ru,sk-de,mk-ta,zh_tw-de,ar-sk,hy-uk,uk-pt,ku-fr_ca,sr-lt,et-my,es-sr,bs-gl,vi-gl,uk-vi,pt_br-az,kk-be,nl-hr,bg-bs,vi-hi,sv-lt,da-mr,cs-zh,fr-kk,it-pl,mn-az,he-sq,ko-ru,ur-hr,ku-nb,pl-id,be-gl,nb-kk,ur-pl,hu-id,he-zh,vi-de,az-sv,fa-hi,pt-bg,bs-eo,de-he,fr-cs,uk-gl,be-bn,mr-lt,hi-fa,mn-zh_tw,tr-lt,cs-zh_tw,nl-hy,sq-eu,da-kk,uk-ar,be-bs,et-mr,hu-ar,mk-ru,de-bn,ro-de,pt_br-mr,pt-sk,lt-sq,eu-he,ms-bn,bg-pt,mn-gl,it-bg,de-id,mr-ko,fr-el,it-eu,ms-pt_br,bg-ur,da-es,az-hu,ro-ta,ms-es,pt-mk,id-gl,hu-bs,ku-hi,he-it,uk-mr,hi-bg,es-hr,pl-uk,es-my,sk-pt,cs-it,my-sk,pl-mk,gl-ro,mr-kk,az-tr,et-hy,bs-az,cs-bs,ko-fr,tr-he,mn-nl,ta-be,hy-hu,zh-ro,es-ko,he-hr,lt-ka,eo-hi,bg-ja,sq-ko,he-ko,nl-zh,bg-uk,ar-gl,hr-nb,my-vi,ku-eu,sr-hy,ms-sk,vi-cs,eu-tr,zh_cn-bn,sk-hy,fi-sq,vi-fr_ca,az-zh_tw,hi-zh_tw,pt_br-ta,fa-id,el-ms,zh_tw-nb,zh_cn-it,sv-pt,el-ro,bs-vi,bg-be,pt_br-ur,sl-sq,fi-fa,zh_cn-be,fr-ta,my-bg,zh-sv,it-ro,it-ms,mk-it,id-hy,th-fa,id-el,kk-fa,hu-he,it-cs,sv-nb,fr-tr,eo-hu,fa-et,my-de,pt-eu,ru-sv,de-ta,ka-ta,my-mk,az-uk,bn-bs,pt_br-ro,pt-et,bn-es,th-sk,pl-eo,my-ku,ku-pt,eo-fi,gl-zh,zh-my,cs-fa,da-mk,fr_ca-zh_tw,vi-mn,mr-de,nb-ja,ja-it,he-nl,be-eo,hi-sq,ku-ro,ko-pl,eo-bn,bn-he,sv-nl,bn-el,it-ja,eo-ja,de-cs,fi-az,gl-lt,id-ur,sk-sl,zh_cn-nb,et-sl,ar-sr,kk-nb,hu-zh_tw,he-zh_cn,mr-mn,hu-gl,fr-hu,ms-ja,sr-th,sv-mr,da-bn,ms-ta,mn-eu,ku-sv,hi-de,he-az,sq-et,zh-ms,ms-tr,mn-hi,it-gl,uk-zh_cn,fr-ro,eu-eo,id-fa,vi-bs,he-lt,et-el,fi-sk,fi-vi,el-ur,eo-zh,ko-bn,de-sr,pl-et,fr-bs,he-fr,hy-ar,tr-ku,cs-uk,vi-sr,sq-nl,my-bs,az-sk,et-ro,fa-es,sr-gl,mn-id,sv-mn,ja-ar,pl-ka,sv-ko,ka-sl,sq-az,eo-kk,gl-ja,da-ar,pl-de,sk-pl,el-pl,ko-be,fa-ko,mn-ku,sv-cs,nb-pt,mk-hr,mr-fa,ar-bs,mr-fi,mk-cs,ur-ja,uk-ru,eo-ms,ru-pl,nb-pt_br,it-de,hr-vi,es-nl,ko-ar,ko-de,kk-pt,hr-hu,da-gl,mn-sk,fr_ca-zh_cn,sr-bs,sq-he,zh-sl,ms-id,ur-hi,ru-sk,bn-hr,pl-sq,hy-mr,tr-kk,bn-kk,pt_br-fr,sr-zh_tw,kk-mn,bn-ja,az-ka,ur-nl,hi-sr,tr-it,ar-ku,fa-bn,it-hr,el-ko,be-da,kk-sk,pt-zh_cn,cs-be,id-he,ja-bn,pt_br-ar,ur-ko,ar-mr,az-hi,ja-sq,tr-th,fi-ar,mr-nb,pt-tr,bg-es,ka-hr,id-zh_tw,uk-hr,ka-sv,ru-fa,de-ms,hi-sk,mn-pt_br,uk-eu,es-zh,id-ta,ja-et,ta-ru,fa-my,be-zh,cs-fr_ca,sl-my,eu-az,cs-he,uk-bs,sv-zh_cn,sk-be,es-ro,ko-nl,pt_br-bn,ta-hr,he-th,ro-et,mk-el,mk-fr,ro-bs,ur-kk,tr-fr,es-ka,de-gl,ro-hy,ar-az,hu-ro,nl-fr,eu-be,fr-bg,sk-ka,gl-hr,ar-bg,fr_ca-uk,fr_ca-be,nb-ms,gl-ur,zh-hi,lt-fa,hi-vi,ku-de,ja-th,az-hy,mr-ku,mk-vi,el-sr,pt_br-zh_cn,he-mr,fi-ro,es-az,de-sq,bn-pl,de-eu,fr_ca-bg,nb-hi,uk-ro,ta-zh_tw,gl-mr,eu-fa,ka-ru,ko-az,ro-ru,eo-gl,pl-tr,ms-el,th-ka,sq-th,lt-kk,ms-fr_ca,az-nb,mr-be,mr-bs,bs-bn,fi-fr_ca,kk-my,sv-he,eo-mn,zh-ru,hr-zh_cn,cs-ku,ro-id,th-ur,fr-ms,ms-hr,mn-sq,hy-nl,eo-az,gl-ko,zh_tw-gl,mn-fr_ca,vi-bn,bs-de,fr_ca-mk,zh_cn-hu,ru-mn,tr-sv,az-ru,el-th,zh-fa,cs-pl,az-bn,cs-ja,th-gl,es-kk,es-zh_cn,bs-fa,zh_cn-es,bg-nl,bg-sv,es-id,hy-fi,pt-fa,sv-hi,nl-ta,it-id,az-ur,gl-cs,bn-ro,fa-pt_br,lt-az,be-nl,ru-eo,bn-lt,ro-pl,he-gl,mr-nl,zh_tw-pt,ur-bn,zh_tw-it,id-nl,ku-hr,hr-bs,tr-ja,ru-bg,id-mk,nb-sl,nl-da,ka-pt,bn-mr,it-fr_ca,nb-my,zh_tw-ur,cs-ta,bg-pl,mr-th,ta-fi,mn-hy,az-es,pt-he,fr_ca-hy,sk-ku,ja-pt_br,zh-ka,sq-ku,nl-pt,ur-el,de-nl,hu-mk,ko-sq,da-sr,fi-tr,mk-sv,hu-bg,ar-ka,sr-ku,hy-hr,el-da,hu-pt_br,fr_ca-mn,hi-et,eo-ku,ur-et,mn-fi,da-pt,sk-bs,sq-el,bg-he,fr_ca-de,ta-hy,es-ms,zh_cn-pt,fa-fi,ku-ms,eu-ro,hu-es,cs-eu,fr_ca-sq,be-ja,ru-it,mr-ur,eo-et,ko-cs,lt-zh_tw,pt-cs,sr-ja,ru-zh_tw,be-lt,sr-el,ta-it,lt-hr,be-hu,ro-fr_ca,lt-el,mr-zh,hu-pl,he-fr_ca,ms-nb,fr-hi,bs-my,pt_br-ko,fa-eu,sv-sk,hr-kk,be-cs,pt-zh_tw,ku-zh_tw,zh-he,sk-th,hi-tr,fa-hu,gl-fi,de-my,ku-be,ur-pt_br,mn-my,ja-id,be-kk,sl-fr_ca,mk-gl,ka-el,sq-mn,ar-cs,id-sk,eu-vi,bs-sl,nb-tr,es-mn,bs-es,sk-fi,eu-th,zh-nb,uk-de,bs-ka,eo-my,be-sk,nl-fr_ca,eu-de,ro-lt,bg-ru,bg-id,id-sr,kk-pt_br,th-ja,lt-pt,mn-es,bg-th,sq-fa,sk-nb,ar-el,pl-gl,bn-az,tr-sr,eu-pt_br,ms-zh,hy-he,ta-mk,tr-fi,ms-be,zh_tw-hi,ar-zh,ta-pl,it-bs,nl-fi,sk-hr,mn-vi,ko-sr,hy-pt,sl-uk,da-sk,et-lt,my-ru,hi-th,fi-ku,ja-ur,tr-hi,pt_br-hy,da-sl,sq-sr,bn-vi,be-pl,sl-gl,gl-hi,tr-bg,et-sv,bs-pl,mn-uk,zh_tw-es,mn-lt,cs-th,mn-et,bs-nb,az-it,bn-fa,ja-es,et-ur,mk-fi,pl-da,de-vi,vi-ka,mn-cs,be-de,zh_cn-mr,ka-id,ko-kk,mk-bs,hy-cs,mn-mk,az-fa,el-zh,fi-ur,sq-da,lt-de,az-mr,bn-sq,ja-be,ur-vi,fi-et,vi-he,fi-kk,uk-fr,mk-ar,ko-gl,ja-fr,ku-th,ru-da,fr-be,nb-lt,uk-cs,tr-da,vi-eo,fr_ca-pt,eo-mr,hy-zh_cn,ku-sq,eo-bs,ta-nb,ro-mn,gl-zh_tw,he-tr,eu-bs,sk-fa,ko-tr,sk-eu,ta-fr_ca,et-fa,hr-pt,id-hi,pt_br-cs,ur-bs,zh-cs,hu-ko,az-ro,ar-mn,uk-kk,de-th,ko-hy,de-ar,he-es,gl-zh_cn,da-et,ka-de,ja-pl,eu-sk,hu-hi,bn-cs,he-cs,hy-eu,kk-eu,he-pl,kk-sv,es-ja,my-sq,nl-pl,zh-eu,my-zh_cn,th-id,gl-be,kk-hr,bs-id,ja-ro,mr-ja,kk-lt,tr-fa,id-fi,pt_br-id,ru-ms,ro-be,hr-pl,ru-fr_ca,zh-sq,ro-th,de-hy,fa-sq,ur-ar,de-kk,th-zh_tw,vi-el,lt-hy,vi-nb,ku-ja,uk-nb,et-ms,da-az,zh-kk,eo-sv,ur-he,fr-sr,pl-pt,ta-zh,nb-gl,es-th,da-vi,pl-kk,ja-tr,ka-ko,ka-hy,fr_ca-kk,zh_cn-th,hr-zh_tw,bs-hi,ro-el,th-vi,eu-ko,pt_br-my,lt-eo,zh-es,ro-hi,mr-cs,sl-ur,mk-nl,pt-fr_ca,bn-my,mk-be,sr-ar,sv-ms,bg-eo,fa-ku,es-sk,cs-pt_br,vi-sq,ta-ms,th-nl,ko-th,eu-cs,th-bg,ja-bg,fr_ca-fr,es-sv,mr-pl,el-mk,my-gl,it-fa,gl-tr,be-et,nb-fr_ca,pt_br-be,es-pl,bg-ar,pt-zh,ar-sl,zh_cn-id,kk-ms,it-tr,ku-sl,fi-gl,sv-hr,zh_tw-mr,ku-ka,be-mr,sl-sk,nb-ur,ro-pt_br,sl-zh,tr-az,nl-ka,ta-pt,ms-bg,ms-ro,he-ar,de-es,eo-sl,fi-eo,nl-th,et-mk,mk-bn,sl-el,ms-mk,ro-sq,zh_cn-ar,da-zh_tw,ja-he,ar-hr,mn-ur,ru-ta,hi-zh_cn,ur-sv,ur-gl,ka-pt_br,az-fr,el-sv,sv-tr,zh_tw-ms,mn-bn,az-id,el-fr_ca,ja-hy,bg-de,kk-el,et-bg,eu-hu,ja-cs,ta-kk,be-ms,sl-fa,uk-my,sk-bn,ta-mr,el-ar,ko-hr,zh_tw-ru,mn-ro,nl-sq,zh_tw-vi,nb-ar,hi-hu,bg-ro,eu-es,es-pt_br,uk-be,ro-hu,id-eu,kk-id,sq-ja,it-bn,cs-fi,ka-zh_cn,bs-bg,ku-ko,be-uk,he-mk,fi-be,ms-uk,ta-ur,cs-nl,ja-kk,he-id,eu-mr,ms-lt,et-fr_ca,ro-es,pl-fr_ca,pl-mn,th-da,he-eo,mk-bg,zh_cn-pl,pl-th,hr-et,zh_tw-sk,mn-sv,zh-eo,zh_tw-el,hr-sr,sr-my,hy-ta,be-ta,lt-ta,mk-he,pt-kk,he-ka,ko-ur,sr-zh,et-cs,mr-ka,pt-bs,he-hi,th-be,sk-ms,nb-sq,ta-et,fi-nb,ar-th,da-my,bn-zh,sk-zh,ja-fi,da-hu,ms-zh_tw,zh-et,bg-hi,gl-sv,es-eu,hy-pt_br,nb-cs,hr-nl,ko-pt,cs-eo,hy-ko,fa-be,mk-pl,sv-th,pt-es,zh_cn-sr,mr-el,vi-zh_cn,eu-mk,it-el,et-bn,bs-ro,gl-ta,bs-pt_br,cs-bn,vi-hr,zh_tw-kk,sk-zh_cn,th-sl,zh_tw-sq,gl-sq,mn-fr,bn-de,eo-bg,el-nb,sv-da,be-sv,zh_cn-he,my-ta,uk-ku,th-zh,id-ar,hu-ru,ar-zh_tw,cs-ms,hu-uk,mr-vi,lt-nl,he-sl,my-bn,bg-zh_tw,hy-da,pt-ms,vi-th,fr-ja,pl-ur,et-ta,kk-fi,ta-uk,nb-uk,nb-sk,he-pt_br,hr-ja,my-pt,mk-th,it-sl,zh_cn-bs,vi-ja,uk-az,sq-zh_cn,tr-mk,ro-fr,sq-kk,be-eu,sq-es,id-vi,it-hu,pl-pt_br,bn-ur,gl-eu,hy-mk,ur-cs,es-bs,gl-az,hr-da,kk-hi,zh_tw-fr,uk-it,cs-zh_cn,fr_ca-cs,ms-sr,my-cs,fi-sr,th-fr_ca,mn-hr,tr-ro,sk-hu,hi-ku,tr-zh,he-bg,fr_ca-zh,hi-id,be-mn,fr-it,zh_cn-sv,zh_cn-ja,sk-hi,mk-de,my-ro,id-ja,sq-mk,mn-kk,de-fi,hi-pt_br,eo-eu,mr-es,uk-th,zh_tw-hr,bs-it,my-el,hr-my,lt-my,et-kk,id-pl,pt_br-pl,my-hr,cs-da,ro-mr,sk-tr,ms-he,ku-az,eu-hy,nl-bg,pt_br-ru,bs-sr,fr-eu,ka-pl,el-nl,sr-de,hu-et,ro-ku,zh-nl,fi-my,et-gl,cs-id,hr-de,ro-sk,uk-tr,vi-sk,ta-sq,sl-pl,ru-ur,hu-eo,fi-bg,eo-ro,he-hu,ro-it,ar-lt,my-eo,sv-az,zh_tw-mn,bg-lt,th-ru,ms-de,bn-sr,es-uk,ku-es,el-eo,bs-ku,nl-cs,ru-hy,my-ko,bs-mk,bg-hy,bg-fa,sr-mk,hi-fi,th-mn,zh_tw-zh,sr-ru,ko-sv,ta-ko,bs-eu,zh-da,id-ku,he-uk,it-sv,fr_ca-id,be-mk,ta-he,zh_cn-hy,pt-pl,fa-pt,be-sl,nb-az,ms-fi,vi-sv,mn-pt,my-tr,be-sq,pt_br-ka,ku-mr,eu-my,fa-fr_ca,pt-eo,sv-de,be-el,mn-fa,hr-ko,fa-sr,da-fi,el-be,hy-sk,sk-pt_br,th-sr,lt-mr,ta-fr,fr-eo,lt-tr,fr_ca-ms,nl-sk,sl-et,hy-zh,pl-fr,sv-pl,mk-sk,uk-pt_br,fa-ms,pt-sr,zh_tw-ar,kk-vi,bn-ta,es-fa,hy-bn,ku-eo,zh-vi,pt-nb,hr-tr,bg-et,fr_ca-eo,az-cs,hi-el,mn-ar,sr-ka,th-hy,vi-hu,ka-th,hi-bs,zh_tw-he,ur-nb,pt_br-nb,bg-sq,da-fr,pt_br-it,zh-tr,hi-az,mk-hi,hr-fi,fr-ru,sl-mk,vi-ar,lt-nb,mk-zh_cn,id-ru,sq-pl,th-fi,hy-be,pl-sr,be-ar,bs-be,az-mk,fr-vi,es-bg,fa-fr,nb-el,de-fa,ja-mn,hu-sk,vi-uk,th-pt,ka-ar,ms-hy,zh_cn-az,pl-fi,vi-it,be-bg,mk-ro,et-uk,gl-el,ur-sl,sq-pt_br,it-zh_tw,fr_ca-he,lt-sr,ta-hi,mn-mr,ku-zh,pt-ku,hi-it,kk-bn,hi-pl,sv-it,hr-ta,lt-bs,sl-eo,sq-de,zh_tw-lt,sq-it,hr-id,eu-sl,hu-it,ru-my,ar-be,ta-bn,bn-id,mn-hu,be-ko,pt_br-sr,pt-pt_br,ta-bg,cs-tr,ka-mn,bn-uk,pl-zh_cn,nb-ta,fr-uk,ka-mk,sl-ja,el-fa,pt_br-tr,cs-mn,es-ku,sq-hr,mn-da,eo-th,fr_ca-my,cs-kk,zh-ja,ko-my,mk-ms,ja-fa,ur-bg,th-ms,sl-hy,sk-fr,bn-eu,zh_cn-eu,ur-ta,bg-hr,fr-th,sv-be,sk-ro,zh-hr,nl-de,sr-pt_br,eo-mk,ka-he,es-sl,hr-el,hr-he,et-ku,ur-fi,cs-sk,nl-az,id-my,ar-uk,sk-da,ja-zh_tw,pl-zh_tw,mr-az,bn-ru,sk-cs,mn-ms,id-th,mn-he,mr-et,fa-it,lt-pl,sv-kk,ar-et,uk-sl,ka-fi,hu-hr,pl-hu,az-fr_ca,az-sr,eu-ms,az-gl,et-ar,nl-bs,eu-ja,az-th,et-fr,mr-ro,nb-hr,mr-pt,it-ur,ku-it,az-he,ja-eu,it-my,uk-bn,fr-zh,ja-sk,tr-be,bs-mn,he-sk,tr-fr_ca,he-de,mn-sl,hu-vi,ka-bs,sk-bg,gl-kk,ko-zh_tw,my-ja,tr-cs,mn-eo,hi-zh,fa-mk,id-bn,nb-mk,nb-es,da-it,vi-sl,sr-mn,hu-hy,de-fr_ca,mr-zh_cn,hi-eo,mr-ar,el-pt_br,ur-tr,fr_ca-ro,kk-gl,sq-bn,ur-zh_cn,hy-fa,hu-ja,ur-eu,pt-id,ur-da,pl-zh,tr-mr,ru-ro,ka-sr,vi-ms,cs-ar,tr-de,mk-nb,fi-pt,ru-bs,bn-fr_ca,el-ru,sr-bg,et-pl,ja-de,vi-mr,sq-fi,tr-ur,zh_cn-mn,sr-hi,ja-hr,eu-hi,zh_cn-vi,ar-ur,th-bn,sl-zh_cn,hr-bg,de-zh_cn,bs-el,my-he,hy-id,he-ta,hy-fr,fa-tr,fr-zh_tw,zh-gl,zh_cn-et,nl-eo,pl-my,ru-mk,eo-uk,el-cs,he-sr,hi-uk,pt_br-ku,th-eu,lt-fi,id-sv,ru-de,pt_br-uk,mk-eo,lt-pt_br,hu-sv,th-mk,ro-eu,ur-hy,ku-cs,fi-it,tr-uk,sv-eu,mn-tr,zh-ko,zh_cn-gl,pt-az,sl-kk,zh-be,az-eu,sv-fi,ms-hi,mk-pt_br,hu-mn,eu-zh,pt_br-bs,ka-az,fi-bs,fr_ca-it,lt-zh_cn,zh_cn-fr,id-et,id-lt,bs-sv,ta-vi,fi-pt_br,nb-mn,et-az,de-zh_tw,id-hr,bs-hu,ko-fr_ca,uk-zh,mk-es,ru-bn,zh_cn-ms,tr-zh_tw,fi-id,et-eu,pl-ko,cs-lt,ja-lt,ar-pl,nl-sl,sv-fa,bn-da,sl-ta,ro-vi,vi-be,lt-sk,hu-th,sq-ta,mr-hr,et-ka,zh-bg,hi-da,zh_cn-zh,ur-mk,ms-sl,de-el,mn-bs,id-ro,ms-eu,mr-tr,hi-mn,sq-tr,bg-cs,el-zh_tw,my-es,he-be,zh_cn-ta,ko-sk,de-da,da-ko,ur-my,ro-fa,bg-fi,my-eu,sv-zh_tw,ku-uk,fa-bg,es-hu,ur-sk,nl-es,pt_br-gl,ka-hu,kk-tr,fa-cs,et-sr,mk-mr,az-sl,el-fi,sr-fr_ca,zh_cn-da,sr-tr,pl-sl,ko-bs,my-nb,tr-pt,fi-ta,mk-id,hy-bs,eo-ta,eo-he,ms-ru,mk-hu,cs-bg,gl-sl,id-zh_cn,nl-ru,ta-lt,gl-he,ko-ms,bn-zh_cn,sk-ta,he-hy,kk-he,cs-el,mn-zh_cn,pl-vi,zh_cn-fr_ca,cs-hy,ru-pt,fa-zh_cn,bg-zh,bn-ms,hi-es,ms-az,sk-mr,es-it,ro-ja,eo-it,hi-hr,da-be,pt_br-th,ar-sv,de-mn,ka-cs,az-zh,gl-nb,fa-sl,sk-az,hy-az,ko-el,ta-ka,fr-et,da-hy,eu-ru,zh_tw-fr_ca,hi-ta,nl-ur,ta-es,fi-mk,pl-ro,lt-ur,fr_ca-fa,uk-ko,bn-fi,ar-my,kk-sr,id-az,sr-mr,eu-fi,fa-sv,ko-zh,be-ur,he-el,da-mn,fr_ca-hi,id-be,ta-fa,nl-it,az-mn,sl-th,fi-sv,bg-ka,kk-ta,ka-mr,et-tr,mk-ko,cs-az,ko-it,hu-fi,nb-bn,be-ru,ta-sr,eo-sq,eo-da,hr-be,da-fr_ca,mk-ja,ur-mr,ku-bg,it-hy,hi-fr_ca,mr-ms,bs-pt,da-ka,ta-cs,uk-et,vi-ro,hy-hi,uk-mk,ur-eo,fa-zh,ku-bn,id-ka,fa-gl,es-cs,fr-my,gl-da,ka-ku,fi-zh_tw,pt_br-hr,fr-fr_ca,sr-sk,sq-ka,ja-mk,sq-nb,id-mn,ka-uk,fr-sl,pl-bs,sv-bg,hu-ur,ru-be,sk-kk,gl-vi,my-lt,ka-lt,pt-sl,sl-fr,mn-ka,pt-ka,ja-bs,et-sq,gl-de,bg-zh_cn,it-eo,da-el,ta-ro,ru-ja,pl-be,th-mr,cs-ka,pl-nb,bg-nb,hi-fr,lt-he,it-et,fr_ca-pt_br,nb-ko,it-mr,cs-sr,nb-eo,ta-ku,ka-fr,hu-kk,hr-mk,pl-ru,ur-sq,eo-ru,et-eo,ka-nl,ru-nb,sr-nl,cs-ko,pt-fr,el-mn,vi-az,sv-fr,ar-pt_br,et-zh,he-fa,el-sq,tr-eo,my-ms,sr-nb,hr-gl,eu-uk,ka-kk,ka-et,ru-pt_br,sq-sl,tr-eu,hy-el,bn-ka,ta-az,id-ko,pt_br-hi,ms-ar,ko-bg,ru-zh_cn,pl-hi,be-th,kk-da,uk-ka,bs-sk,nb-he,kk-ko,pt-uk,vi-my,bg-it,el-eu,mk-sq,ja-hi,el-ka,vi-nl,my-sv,tr-sl,ku-he,et-fi,vi-et,sq-pt,zh-sk,sq-id,hy-mn,ru-th,az-ta,he-ku,nb-bs,pt_br-eu,ms-bs,el-pt,sl-eu,nb-fr,et-ru,id-cs,bn-th,kk-fr_ca,zh_tw-zh_cn,ar-sq,hy-ro,cs-hi,pt_br-el,fr_ca-vi,ku-gl,hr-eu,pl-bn,et-mn,kk-ar,tr-ta,sr-be,sv-ka,lt-vi,eu-fr,he-nb,ru-ku,pt_br-fa,hy-zh_tw,ru-sq,kk-ru,de-tr,ro-az,fa-mr,fr-sk,ko-he,hr-ur,ja-nb,lt-ms,es-et,fa-ur,it-uk,nb-sv,fa-hr,el-hr,sv-hu,hi-gl,zh-bs,hr-zh,bn-ko,gl-hy,el-vi,hy-lt,uk-bg,hr-pt_br,my-be,nl-nb,de-az,hi-eu,sk-uk,ku-pl,bs-fr_ca,ko-ro,gl-th,sk-eo,az-kk,zh_tw-ja,es-vi,zh-bn,eu-nb,kk-de,et-nl,bn-it,zh_tw-pl,bn-nb,uk-lt,vi-fa,de-pt_br,ka-eo,tr-el,ru-mr,ja-pt,sl-ro,my-sr,ro-zh,ur-ro,fr-mr,ar-de,uk-nl,nb-be,sv-el,zh_tw-fi,my-fi,bg-ms,ar-da,vi-pt_br,th-he,sl-sv,sl-de,hi-be,id-hu,ru-lt,az-my,ms-ur,eo-fa,sr-pt,az-zh_cn,nl-ja,sq-ar,pt_br-es,da-nl,zh_cn-zh_tw,be-zh_tw,gl-sr,et-vi,th-tr,ka-gl,pl-es,ru-hu,nl-pt_br,lt-it,az-hr,zh-pt_br,ko-hi,zh-ar,fa-nl,eo-lt,bs-he,da-cs,th-cs,mk-lt,cs-vi,pt_br-bg,fa-pl,nl-gl,hu-mr,ku-mn,pt_br-lt,lt-uk,zh_cn-bg,mk-hy,sq-hy,hu-ms,az-ar,ru-he,be-it,bg-fr,my-zh,de-uk,be-hy,pt_br-vi,lt-ja,it-zh_cn,ms-fr,de-nb,mr-my,bs-zh_tw,el-ku,sv-ro,uk-mn,es-hy,tr-et,ar-nl,fr_ca-ja,uk-eo,sv-sq,zh_tw-ro,zh_tw-pt_br,gl-mk,sl-he,pt_br-sq,ku-tr,kk-mk,ka-sk,ja-sr,bg-fr_ca,uk-sq,ka-it,it-he,ru-eu,nb-mr,mn-ru,id-de,ja-mr,ur-th,da-eu,fr_ca-ur,id-fr,ko-mk,eo-pl,zh_tw-id,et-sk,et-be,id-pt,az-pl,pl-nl,pt-it,ur-pt,ms-th,pl-hr,eo-nl,id-bg,he-kk,fr_ca-eu,eo-pt,uk-hu,sr-hr,ka-hi,ku-hy,es-ru,pt-da,pl-ta,sr-bn,hi-ru,nb-nl,cs-es,el-lt,ms-et,ko-uk,pt-el,id-sq,sk-ur,sk-sr,mk-tr,nb-eu,hu-nl,fr_ca-ko,it-hi,zh_tw-ku,zh-de,zh_cn-eo,nb-th,ru-gl,es-de,lt-eu,sv-et,sq-sk,hu-sq,et-da,bg-tr,zh-el,bn-mk,hr-bn,ku-id,mr-it,az-ja,sk-mn,ur-be,es-fr_ca,kk-sl,sk-ja,pt_br-hu,mr-sl,id-da,id-tr,et-pt,sr-hu,fi-zh_cn,bn-fr,mr-pt_br,bg-el,hy-sr,ka-fr_ca,eu-sr,pt_br-sl,pt_br-pt,pt_br-ms,ar-tr,el-sl,hr-ro,fr-pt_br,sl-hi,eo-nb,eo-be,pl-cs,hu-fa,vi-pt,az-bg,th-hu,ka-ro,ka-bg,ro-ms,th-ku,sv-mk,ru-vi,bn-hy,ms-cs,sr-ro,th-lt,da-th,ta-bs,id-mr,nl-he,fi-da,gl-pt,pt_br-nl,it-zh,zh-id,da-hr,fa-vi,mr-fr_ca,cs-ro,ur-az,hy-ru,zh-ta,ta-gl,ka-be,fr_ca-ru,sv-ku,el-hi,eo-id,ta-ar,bs-ur,fi-el,vi-eu,th-zh_cn,ms-sq,eo-vi,es-ta,da-uk,mn-ta,ru-hr,ja-da,kk-zh_cn,ko-zh_cn,it-ru,vi-ta,ru-ka,de-sk,zh_tw-sr,zh_cn-ka,bn-ku,fa-nb,zh_tw-cs,kk-et,da-nb,ar-fi,fa-ta,bs-tr,sl-lt,he-my,el-bg,pt-ru,ja-hu,nl-tr,nl-ko,sq-fr,de-it,ja-sv,es-fi,it-fr,tr-ar,cs-mk,fr-es,de-pt,eo-el,eo-fr_ca,hu-az,mr-bg,pl-el,hi-mr,ru-ar,hr-es,zh_tw-th,hy-nb,cs-de,hr-az,hu-el,eu-ka,my-mr,pt-ko,zh_tw-be,da-ms,nl-ro,ta-tr,he-vi,sr-sv,ms-ku,ro-nb,vi-ru,sl-hu,be-fr,gl-bg,zh_tw-my,de-ja,da-eo,et-bs,sl-cs,my-fa,nb-ka,eo-cs,sq-ms,mr-sr,ur-de,az-et,mk-az,hu-ku,hi-ro,sv-fr_ca,zh_cn-tr,hr-ms,ru-kk,es-sq,bs-kk,ms-nl,uk-el,kk-eo,hu-bn,nl-uk,pt-hu,ko-mr,fi-lt,pt-ja,my-ka,ar-ms,da-pt_br,cs-hu,bn-bg,ta-hu,zh_cn-ur,nb-hu,ta-nl,sr-ms,ro-da,fr-az,da-hi,sk-lt,fr-id,id-sl,ro-ar,fi-sl,et-nb,eu-bn,my-ar,sr-ko,ka-fa,az-de,fr-he,zh-fi,hu-zh,hi-nl,sq-my,pt-lt,ru-fi,zh_cn-mk,cs-nb,fr_ca-th,lt-bg,bs-et,eu-pt,ja-eo,ar-zh_cn,de-bs,sk-vi,ro-pt,ko-da,ja-ru,gl-it,pt_br-et,it-es,lt-ku,bs-th,mr-sk,bn-sl,sv-id,hy-es,cs-sv,kk-es,de-ur,ro-hr,pl-he,ur-ms,et-de,zh_cn-sl,zh_tw-sv,zh-hy,ru-sl,mr-ru,ur-ka,sk-my,fi-ja,ur-fr_ca,fr-hy,de-ru,lt-th,mn-pl,sv-sr,sk-ar,it-da,ja-ms,hi-sl,mr-sv,sl-vi,hi-sv,sq-cs,ru-uk,th-de,hy-sv,th-sv,hi-bn,nb-ru,fa-de,bg-pt_br,hu-da,ku-mk,ro-zh_cn,ja-el,it-kk,eo-hy,fr-fi,pt_br-mk,bg-kk,zh_cn-nl,hi-ja,fr_ca-gl,ko-sl,ku-ar,hu-eu,ms-fa,ta-sv,zh-zh_tw,bs-ta,gl-et,pt_br-eo,de-pl,es-hi,he-da,ur-it,fi-cs,fr_ca-ta,th-bs,de-hu,bn-be,nl-lt,kk-ro,ta-eu,fr_ca-az,zh_cn-ku,my-hu,kk-zh_tw,zh-pl,my-id,hy-th,pt_br-mn,az-ku,nb-id,tr-bn,my-it,nl-id,ro-sv,my-nl,kk-fr,el-sk,vi-fi,id-nb,de-hi,it-ta,th-ta,hr-hi,bg-gl,kk-sq,ar-it,hi-hy,hr-lt,az-bs,tr-ka,cs-fr,ko-vi,tr-sq,ru-ko,bs-ja,de-zh,gl-mn,es-ar,hu-tr,pl-bg,fr-sq,ro-cs,fr-hr,zh-uk,es-be,zh_tw-hy,sk-ru,mr-zh_tw,tr-sk,nl-sv,es-da,he-sv,pl-mr,ur-hu,sk-zh_tw,be-nb,ar-pt,ko-hu,mk-fr_ca,sv-bn,fr_ca-tr,bs-hr,de-ro,ur-sr,eo-es,ms-hu,gl-es,uk-ta,lt-hi,nl-eu,hr-ru,et-es,hi-cs,ja-fr_ca,hu-ka,sl-ru,ta-da,pt-gl,nl-et,fi-hr,de-sl,ta-ja,ja-zh,nl-mk,el-he,tr-vi,mk-zh,sr-fa,sr-da,th-et,pl-eu,cs-my,ko-nb,sl-az,fa-da,th-ar,mk-fa,zh_cn-fi,mk-zh_tw,uk-es,de-lt,ka-nb,et-it,he-ur,fa-mn,sr-it,mr-id,gl-nl,ru-fr,cs-hr,bs-zh_cn,uk-pl,ja-vi,ar-id,bs-uk,kk-az,et-ja,it-ka,th-fr,sr-ur,be-vi,fi-nl,ar-vi,sv-bs,nb-ku,ta-id,de-ka,eu-bg,nb-bg,fr_ca-fi,my-pt_br,kk-zh,nl-vi,my-sl,sr-sl,da-ja,sl-mn,sv-ta,ja-ta,mr-fr,es-bn,nb-hy,ko-ta,sr-vi,fr-zh_cn,bs-da,ro-kk,et-pt_br,fi-uk,ms-eo,lt-hu,sl-nl,uk-fa,nl-ms,de-be,uk-da,az-lt,it-pt_br,mn-zh,sl-ms,mr-hy,bn-sv,fr_ca-el,sl-pt_br,fa-el,my-et,sk-et,cs-sl,fr-mk,eu-el,el-bs,ja-ka,it-sk,zh_tw-eo,vi-bg,mn-be,el-kk,ta-sl,zh_cn-my,ur-id,fi-zh,uk-sk,ka-bn,ko-fa,fr-lt,lt-sv,vi-pl,tr-ko,kk-hy,fa-zh_tw,vi-da,zh_cn-uk,fi-hy,kk-uk,hr-ar,ku-kk,el-my,gl-eo,sl-fi,ms-zh_cn,ta-eo,bs-ms,he-eu,gl-ms,my-fr_ca,bg-ku,mn-nb,ms-pl,ko-eo,ru-sr,es-eo,hr-cs,de-sv,ro-sr,hi-nb,eo-zh_tw,et-he,da-bs,fa-eo,ur-es,nb-de,nl-bn,id-kk,sv-zh,th-hi,pl-fa,pl-ar,nb-zh,gl-id,zh_tw-da,nb-zh_tw,sr-he,nb-et,ro-zh_tw,kk-hu,tr-pl,hy-kk,gl-pl,fa-bs,bs-ar,da-zh,nl-ku,fr_ca-bn,hu-be,eu-gl,hr-mn,zh_cn-pt_br,de-ko,it-pt,ja-my,bs-zh,ka-da,ka-es,hy-sq,mn-el,fa-az,ru-es,sv-es,hy-pl,lt-id,zh-pt,cs-ru,ru-nl,sk-nl,eu-zh_tw,pt-bn,fi-he,fa-kk,ms-vi,hr-sq,sl-it,mk-pt,ro-gl,bs-fi,nb-ro,da-ur,fr-sv,nl-zh_cn,uk-hy,sl-ka,fr-ka,el-fr,sk-sv,zh_cn-cs,it-nl,ta-de,es-gl,zh_cn-lt,ms-gl,sl-ar,ku-lt,hu-zh_cn,th-my,ka-my,ka-vi,sr-pl,zh_tw-az,tr-ru,pt-ur,mr-uk,az-sq,sv-eo,ru-zh,el-id,be-hr,pt-mr,eu-ku,fa-he,mr-gl,ja-zh_cn,he-bn,hy-sl,hi-kk,be-hi,eu-et,mn-ko,sr-es,ko-mn,da-pl,vi-ur,vi-ko,bg-da,hu-nb,bs-hy,mk-sl,th-eo,lt-fr,fr-nl,bg-vi,da-de,bn-zh_tw,nl-my,fr-mn,sq-ru,hr-ka,pt_br-fi,tr-nb,zh_tw-fa,fi-bn,sl-bg,hy-ja,kk-pl,sk-id,cs-gl,ko-ku,gl-fa,ko-pt_br,mk-uk,cs-mr,zh_cn-ko,sl-da,ro-he,hy-et,sl-pt,zh_tw-nl,tr-my,hr-eo,mk-ku,zh-az,el-et,cs-sq,hr-uk,sq-vi,eo-ka,zh-mr,mn-it,ms-pt,uk-he,fr_ca-hr,th-az,be-he,hy-ku,ku-nl,hi-he,sv-my,it-be,bn-sk,de-mr,hy-my,et-hi,bs-lt,vi-ku,he-pt,zh_tw-ka,bs-ru,et-zh_tw,sl-nb,zh_cn-kk,sq-hi,zh_tw-sl,id-pt_br,it-lt,ku-pt_br,hy-tr,pl-lt,bn-hi,mr-da,nb-pl,sv-ar,lt-da,sr-eo,sk-es,zh-fr,pt-hy,ku-fi,gl-bs,zh-fr_ca,gl-pt_br,zh_cn-hr,nl-hu,kk-th,ko-fi,ta-my,ta-zh_cn,lt-mn,da-ru,ja-uk,mk-mn,da-id,ru-cs,gl-uk,ro-tr,be-ka,az-pt_br,sl-bs,vi-hy,fa-hy,sq-zh_tw,mr-hu,sq-ur,uk-fr_ca,ka-tr,it-mk,et-zh_cn,zh_tw-bn,sk-mk,da-ku,el-uk,fa-ro,be-id,mr-eu,uk-sv,fr_ca-pl,id-uk,sl-be,sk-he,ro-bn,az-ko,fr-da,bn-pt_br,zh-lt,ar-fr,ku-da,pl-hy,zh_cn-el,nl-mr,ro-mk,ko-id,ar-kk,mk-ka,lt-cs,he-et,kk-bg,bg-mk,sr-sq,fr-ar,tr-gl,nb-it,sv-vi,eo-sr,ja-nl,bs-sq,zh-it,mk-sr,ta-th,lt-et,ms-sv,pt_br-da,zh_tw-mk,ro-bg,hr-th,pt_br-fr_ca,bs-ko,es-el,kk-ku,vi-zh_tw,hi-lt,lt-ko,pl-az,zh_tw-ta,eo-hr,sq-fr_ca,ru-hi,ka-zh,hr-it,mr-hi,fr_ca-ka,el-mr,th-sq,kk-mr,sr-fr,es-zh_tw,id-it,ku-ta,ar-eu,nb-zh_cn,bs-nl,tr-es,es-mk,pl-ms,sv-ur,th-es,hy-de,sq-uk,he-mn,zh-zh_cn,eu-sq,sv-sl,it-th,be-zh_cn,be-az,ur-ru,tr-id,ur-uk,bg-ko,hy-bg,ms-it,fr-fa,es-nb,fa-uk,eu-zh_cn,sk-it,ms-da,gl-fr,fa-th,be-pt_br,ms-kk,da-sv,ja-ko,my-hi,ar-ja,fi-hu,ja-gl,de-bg,it-ku,sl-es,pt-ta,fa-ar,bn-mn,fi-eu,sl-id,id-es,th-pt_br,sl-hr,ka-sq,mk-kk,be-sr,cs-pt,az-nl,hy-eo,ro-ka,ro-uk,fr-ko,fr_ca-lt,pt-th,be-fi,el-zh_cn,fr_ca-sr,my-fr,pt-sv,it-ko,th-kk,zh-sr,hu-lt,fr_ca-nb,mk-da,eo-sk,ku-bs,pt-be,ja-sl,eu-ar,tr-nl,fa-ka,mn-th,ka-ur,sq-be,zh-th,es-tr,zh_cn-fa,ta-sk,be-pt,th-el,gl-my,ro-ur,vi-tr,el-tr,nl-fa,ar-hy,hu-sr,sr-kk,hr-sl,ja-az,el-de,hu-sl,tr-hy,eo-zh_cn,my-uk,hi-pt,de-ku,ku-vi,bg-sr,fi-ru,pt_br-zh,bn-et,bn-gl,nb-vi,bs-fr,fr_ca-es,ru-tr,fa-ja,mr-ta,vi-id,el-az,de-mk,ka-eu,et-th,mk-my,fr-ur,bg-mn,sq-bs,be-es,eo-ar,sk-ko,ro-ko,az-fi,lt-be,da-sq,nl-sr,eu-it,et-ko,my-hy,bg-mr,nl-zh_tw,de-hr,ta-pt_br,gl-ka,hy-fr_ca,pt_br-sk,sr-et,fr_ca-sk,ko-ja,de-et,kk-nl,bs-cs,tr-ms,uk-fi,hi-ka,tr-bs,fi-fr,ar-eo,lt-es,zh_cn-de,be-fr_ca,ms-ka,hr-fr,fr_ca-sl,pt-my,id-eo,et-hu,sq-lt,zh_tw-eu,zh_tw-bg,kk-cs,et-hr,hi-ms,de-fr,my-ur,it-nb,ku-fa,he-ru,fa-ru,zh-mn,bg-bn,th-hr,bn-pt,mr-he,hu-pt,hu-my,hr-mr,mn-sr,hi-ar,hu-cs,sv-ru,my-pl,pt-nl,nl-mn,ur-mn,sl-zh_tw,ko-et,fi-hi,nl-be,lt-fr_ca,my-az,mn-de,ja-ku,ar-hu,be-ku,ar-es,lt-ar,lt-sl,da-fa,de-eo,gl-sk,mr-bn,sq-zh,my-zh_tw,mn-ja,gl-ku,eu-fr_ca,es-mr,tr-mn,ms-mn,pt-hr,he-zh_tw,kk-it,id-ms,hu-fr,uk-id,fi-pl,sr-ta,zh_cn-ru,bg-sl,pt-ar,my-mn,fr-de,nl-hi,kk-ja,id-fr_ca,pt_br-he,my-da,az-da,bn-tr,hi-mk,ku-sr,pt-ro,pt-de,da-ro,ku-et,eo-fr,lt-zh,uk-ur,sv-ja,he-bs,sl-tr,kk-bs,ku-my,eo-ko,nb-fa,el-hy,uk-sr,hi-my,fr-gl,es-pt"

checkpoint="the best checkpoint selected according to the validation set on supervised translation directions"

zero_thot_translation_dir=${zero_shot_translation_output_dir}/${checkpoint}
for lang_pair in ${lang_pairs//,/ }; do
    array=(${lang_pair//-/ })
    src_lang=${array[0]}
    tgt_lang=${array[1]}
    parallel_trans_dir=${zero_thot_translation_dir}/${lang_pair}

    options="-l "
    if [[ ${tgt_lang:0:2} = "zh" ]]; then
        options+="${src_lang}-${tgt_lang:0:2}"
    else
        options+="${src_lang}-${tgt_lang}"
    fi
    
    sacremoses -l ${tgt_lang:0:2} detokenize < ${parallel_trans_dir}/sys.test.${src_lang}-${tgt_lang}.${tgt_lang} > ${parallel_trans_dir}/sys.test.detok.${src_lang}-${tgt_lang}.${tgt_lang}

    score=$(sacrebleu -w 6 ${options} ${raw_zero_shot_corpus_dir}/${lang_pair}/test.detok.${src_lang}-${tgt_lang}.${tgt_lang} < ${parallel_trans_dir}/sys.test.detok.${src_lang}-${tgt_lang}.${tgt_lang})
    echo ${checkpoint}.${lang_pair}.${score}
done
