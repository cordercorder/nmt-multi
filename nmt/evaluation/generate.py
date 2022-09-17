import ast
import logging
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from omegaconf import DictConfig

from nmt.data_handling import write_data


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    def get_task(lang_pair: str):
        source_lang, target_lang = lang_pair.split("-")
        task_config = Namespace(**vars(cfg.task))
        task_config.source_lang = source_lang
        task_config.target_lang = target_lang
        return tasks.setup_task(task_config)

    # all source languages should share dictionary
    # all target languages should share dictionary
    evaluation_lang_pairs = cfg.task.evaluation_lang_pairs.split(",")

    lang_pair = evaluation_lang_pairs[0]
    task = get_task(lang_pair)

    # task = tasks.setup_task(cfg.task)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    # the model is shared between all language pairs
    # use the first task to load model
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    translation_output_dir = cfg.task.translation_output_dir
    translation_file_formart = cfg.task.translation_file_formart

    for i, lang_pair in enumerate(evaluation_lang_pairs):
        # start translation process
        parallel_translation_dir = os.path.join(translation_output_dir, lang_pair)

        os.makedirs(parallel_translation_dir, exist_ok=True)
        source_lang, target_lang = lang_pair.split("-")

        translation_file_path = os.path.join(parallel_translation_dir, translation_file_formart.format(cfg.dataset.gen_subset, source_lang, target_lang, target_lang))
        if os.path.isfile(translation_file_path) and not cfg.task.force_replace_old_translation:
            # has been translated!
            logger.info(f"Retranslation {translation_file_path}")
            continue
        # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
        # the saved_cfg.task is language independent, so it can be share across language pairs
        # load dataset one by one to reduce memory requirement
        if i > 0:
            task = get_task(lang_pair)

        task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)
        # Load dataset (possibly sharded)
        itr = task.get_batch_iterator(
            dataset=task.dataset(cfg.dataset.gen_subset),
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(), *[m.max_positions() for m in models]
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=cfg.distributed_training.distributed_world_size,
            shard_id=cfg.distributed_training.distributed_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

        for model in models:
            if hasattr(model.decoder, "set_target_lang"):
                model.decoder.set_target_lang(target_lang)
                assert hasattr(model.decoder, "target_lang")

        # Initialize generator

        extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
        generator = task.build_generator(
            models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

        # Handle tokenization and BPE
        tokenizer = task.build_tokenizer(cfg.tokenizer)
        bpe = task.build_bpe(cfg.bpe)

        def decode_fn(x):
            if bpe is not None:
                x = bpe.decode(x)
            if tokenizer is not None:
                x = tokenizer.decode(x)
            return x

        translations = []
        has_target = True
        for sample in progress:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if "net_input" not in sample:
                continue
            
            if cfg.task.add_tgt_lang_ids_to_sample_input:
                sample["net_input"]["tgt_lang_id"] = sample.get("tgt_lang_id", None)

            prefix_tokens = None
            if cfg.generation.prefix_size > 0:
                prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

            constraints = None
            if "constraints" in sample:
                constraints = sample["constraints"]

            hypos = task.inference_step(
                generator,
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
            )

            for i, sample_id in enumerate(sample["id"].tolist()):
                has_target = sample["target"] is not None

                # Remove padding
                if "src_tokens" in sample["net_input"]:
                    src_tokens = utils.strip_pad(
                        sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                    )
                else:
                    src_tokens = None

                target_tokens = None
                if has_target:
                    target_tokens = (
                        utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                    )

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(cfg.dataset.gen_subset).src.get_original_text(
                        sample_id
                    )
                    target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(
                        sample_id
                    )
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(
                            target_tokens,
                            cfg.common_eval.post_process,
                            escape_unk=True,
                            extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                                generator
                            ),
                        )

                src_str = decode_fn(src_str)
                if has_target:
                    target_str = decode_fn(target_str)

                # Process top predictions
                for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=src_str,
                        alignment=hypo["alignment"],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=cfg.common_eval.post_process,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                    )
                    detok_hypo_str = decode_fn(hypo_str)
                    translations.append((detok_hypo_str, sample_id))
        translations = [detok_hypo_str for detok_hypo_str, sample_id in sorted(translations, key=lambda item: item[1])]
        write_data(translations, translation_file_path)


def add_personal_args(parser):
    parser.add_argument("--evaluation-lang-pairs", type=str, required=True)
    parser.add_argument("--translation-output-dir", type=str, required=True)
    parser.add_argument("--translation-file-formart", type=str, default="sys.{}.{}-{}.{}")
    parser.add_argument("--add-tgt-lang-ids-to-sample-input", default=False, action="store_true")
    parser.add_argument("--force-replace-old-translation", action="store_true")


def cli_main():
    parser = options.get_generation_parser()
    add_personal_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
