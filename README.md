# Informative Language Representation Learning for Massively Multilingual Neural Machine Translation.

[**Paper**](https://aclanthology.org/2022.coling-1.458/) |
[**Requirements**](#Requirements) |
[**Fairseq Installation**](#Fairseq-Installation) |
[**Data Preprocessing**](#Data-Preprocessing) |
[**Model Training**](#Model-Training) |
[**Evaluation**](#Evaluation) |
[**Citation**](#Citation)


## Requirements
 - Python >= 3.7
 - PyTorch >= 1.6.0
 - Fairseq with [commit ID d3890e5](https://github.com/facebookresearch/fairseq/tree/d3890e593398c485f6593ab8512ac51d37dedc9c)
 - sacrebleu < 2.0


## Fairseq Installation
We build the multilingual neural machine translation models based on Fairseq library. Please install it first:
```bash
cd fairseq_dir
pip install -e ./
```

## Data Preprocessing
There are four steps in the data preprocessing pipeline:

1. BPE model training
2. Subword segmentation
3. Removing long sentences
4. Binarizing the data with `fairseq-preprocess`

We provide script examples to run the pipeline described above for preprocessing the parallel corpus of multiple language pairs:
```
scripts/ted/data_process/multilingual_preprocess.sh
scripts/opus-100/data_process/multilingual_preprocess.sh
```

There is an artificial English tag \_\_en\_\_ prepended to every non-English sentence in the raw TED-59 dataset, which may affect model training and bias BLEU. Please execute `scripts/ted/data_process/remove_start_token.sh` to remove this tag before the preprocessing pipeline for TED-59 dataset (`scripts/ted/data_process/multilingual_preprocess.sh`).


**Note that please insert the path of `nmt-multi` directory to the [inport path](https://docs.python.org/3/glossary.html#term-import-path) before executing these shell scripts to make Python interpreter aware of the `nmt` package.** There is an example below:
``` bash
# modify the environment variable PYTHONPATH
export PYTHONPATH=/path/to/nmt-multi:${PYTHONPATH}
```


## Model Training
Once the data has been preprocessed, the multilingual neural machine translation models can be trained with the shell scripts in the `scripts/opus-100/train` and `scripts/ted/train` folders. Note that please set the variables in these scripts properly before executing them:
```bash
bash scripts/opus-100/train/fairseq_train.many-many.laa.sh
```
The training for other models (e.g., token_src, token_tgt, lee) can be done in similar way, please refer `scripts/opus-100/train` and `scripts/ted/train` folders for more details.

## Evaluation
The evaluation pipeline is composed of three steps:

1. Translate the validation sets (only for supervised language pairs) with the saved checkpoints
```bash
bash scripts/opus-100/evaluation/eval.valid.many-many.laa.sh
```


2. Select the best checkpoint according to the average BLEU on the validation sets
```bash
# calculate BLEU score
bash scripts/opus-100/evaluation/report_bleu.valid.many-many.laa.sh > report_bleu.valid.many-many.laa.logs

# convert report_bleu.valid.many-many.laa.logs into json format
# --input denotes the path of report_bleu.valid.many-many.laa.logs
# --output_json_data denotes the path of the output json file
bash scripts/opus-100/evaluation/multilingual_bleu_statistics.sh

# report the average BLEU score for each checkpoint
# the checkpoints will be printed in descending order of average BLEU
# the checkpoint with the highest average BLEU was selected as the best checkpoint in our work
bash scripts/opus-100/evaluation/get_best_checkpoint.sh
```


3. Translate the test sets with the selected checkpoint for both supervised and zero-shot translation

For supervised translation:
```bash
# translate the test sets of supervised language pairs
bash scripts/opus-100/evaluation/eval.test.many-many.laa.sh

# calculate BLEU score
bash scripts/opus-100/evaluation/report_bleu.test.many-many.laa.sh > report_bleu.test.many-many.laa.logs

# convert report_bleu.test.many-many.laa.logs into json format
# --input denotes the path of report_bleu.test.many-many.laa.logs
# --output_json_data denotes the path of the output json file
bash scripts/opus-100/evaluation/multilingual_bleu_statistics.sh
```

For zero-shot translation
```bash
# translate the test sets of zero-shot language pairs
bash scripts/opus-100/evaluation/eval.zero-shot.many-many.laa.sh

# calculate BLEU score
bash scripts/opus-100/evaluation/report_bleu.zero-shot.many-many.laa.sh > report_bleu.zero-shot.many-many.laa.logs

# convert report_bleu.zero-shot.many-many.laa.logs into json format
# --input denotes the path of report_bleu.zero-shot.many-many.laa.logs
# --output_json_data denotes the path of the output json file
bash scripts/opus-100/evaluation/multilingual_bleu_statistics.sh
```


The evaluation for other models (e.g., token_src, token_tgt, lee) can be done in similar way, please refer `scripts/opus-100/evaluation` and `scripts/ted/evaluation` folders for more details.


## Citation

```
@inproceedings{DBLP:conf/coling/JinX22,
  author    = {Renren Jin and
               Deyi Xiong},
  editor    = {Nicoletta Calzolari and
               Chu{-}Ren Huang and
               Hansaem Kim and
               James Pustejovsky and
               Leo Wanner and
               Key{-}Sun Choi and
               Pum{-}Mo Ryu and
               Hsin{-}Hsi Chen and
               Lucia Donatelli and
               Heng Ji and
               Sadao Kurohashi and
               Patrizia Paggio and
               Nianwen Xue and
               Seokhwan Kim and
               Younggyun Hahm and
               Zhong He and
               Tony Kyungil Lee and
               Enrico Santus and
               Francis Bond and
               Seung{-}Hoon Na},
  title     = {Informative Language Representation Learning for Massively Multilingual
               Neural Machine Translation},
  booktitle = {Proceedings of the 29th International Conference on Computational
               Linguistics, {COLING} 2022, Gyeongju, Republic of Korea, October 12-17,
               2022},
  pages     = {5158--5174},
  publisher = {International Committee on Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.coling-1.458},
  timestamp = {Thu, 13 Oct 2022 17:29:38 +0200},
  biburl    = {https://dblp.org/rec/conf/coling/JinX22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
