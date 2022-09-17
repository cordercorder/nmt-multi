import argparse
import math

from argparse import Namespace
from nmt.data_handling import load_json, save_json


def _get_win_rate(args: Namespace):
    baseline_data = load_json(args.baseline)
    others = load_json(args.others)
    # validate data in baseline
    # baseline_data and other format: {"checkpoint": {"lang_pair1": bleu, "lang_pair2": bleu}}
    baseline_lang_pairs = set(baseline_data[next(iter(baseline_data))].keys())
    for bleu_data in baseline_data.values():
        _tmp_lang_pairs = set(bleu_data.keys())
        assert len(baseline_lang_pairs - _tmp_lang_pairs) == 0, (
            f"baseline_lang_pairs: {baseline_lang_pairs}, _tmp_lang_pairs: {_tmp_lang_pairs}"
        )

    def validata_data(_others):
        baseline_checkpoints = set(baseline_data.keys())
        _others_checkpoints = set(_others.keys())
        if args.validate_checkpoints:
            assert len(baseline_checkpoints - _others_checkpoints) == 0, (
                f"baseline_checkpoints: {baseline_checkpoints}, _tmp_lang_pairs: {_others_checkpoints}"
            )
        for bleu_data in _others.values():
            _tmp_lang_pairs = set(bleu_data.keys())
            assert len(baseline_lang_pairs - _tmp_lang_pairs) == 0, (
            f"baseline_lang_pairs: {baseline_lang_pairs}, _tmp_lang_pairs: {_tmp_lang_pairs}"
        )
    
    validata_data(others)

    baseline_checkpoint = None
    if len(baseline_data) == 1 and len(others) == 1:
        baseline_checkpoint = next(iter(baseline_data.keys()))

    result = {}
    pivot_lang = getattr(args, "pivot_lang", None)

    for checkpoint, bleu_data in others.items():
        win_rate = 0
        max_bleu_gain = -math.inf
        min_bleu_gain = math.inf

        max_bleu_gain_from_pivot = -math.inf
        min_bleu_gain_from_pivot = math.inf
        max_bleu_gain_to_pivot = -math.inf
        min_bleu_gain_to_pivot = math.inf

        num_langs_from_pivot = None
        num_langs_to_pivot = None

        if pivot_lang is not None:
            num_langs_from_pivot = 0
            num_langs_to_pivot = 0

        win_rate_from_pivot = None
        win_rate_to_pivot = None
        if pivot_lang is not None:
            win_rate_from_pivot = 0
            win_rate_to_pivot = 0

        for lang_pair, bleu_point in bleu_data.items():
            baseline_bleu_point = baseline_data[baseline_checkpoint][lang_pair] if baseline_checkpoint is not None else baseline_data[checkpoint][lang_pair]
            bleu_gain = bleu_point - baseline_bleu_point

            src_lang, tgt_lang = lang_pair.split("-")

            if pivot_lang is not None:
                if src_lang == pivot_lang:
                    num_langs_from_pivot += 1
                if tgt_lang == pivot_lang:
                    num_langs_to_pivot += 1

            if bleu_point > baseline_bleu_point:
                win_rate += 1

                max_bleu_gain = max(max_bleu_gain, bleu_gain)

                if pivot_lang is not None:
                    if src_lang == pivot_lang:
                        win_rate_from_pivot += 1
                        max_bleu_gain_from_pivot = max(max_bleu_gain_from_pivot, bleu_gain)

                    elif tgt_lang == pivot_lang:
                        win_rate_to_pivot += 1
                        max_bleu_gain_to_pivot = max(max_bleu_gain_to_pivot, bleu_gain)
            else:
                min_bleu_gain = min(min_bleu_gain, bleu_gain)
                if pivot_lang is not None:
                    if src_lang == pivot_lang:
                        min_bleu_gain_from_pivot = min(min_bleu_gain_from_pivot, bleu_gain)

                    elif tgt_lang == pivot_lang:
                        min_bleu_gain_to_pivot = min(min_bleu_gain_to_pivot, bleu_gain)


        if pivot_lang is not None:
            assert win_rate == win_rate_from_pivot + win_rate_to_pivot
            assert num_langs_to_pivot + num_langs_from_pivot == len(baseline_lang_pairs)
            assert min_bleu_gain == min(min_bleu_gain_from_pivot, min_bleu_gain_to_pivot)
            assert max_bleu_gain == max(max_bleu_gain_from_pivot, max_bleu_gain_to_pivot)

            win_rate_from_pivot = win_rate_from_pivot / num_langs_from_pivot
            win_rate_to_pivot = win_rate_to_pivot / num_langs_to_pivot
        
        win_rate = win_rate / len(baseline_lang_pairs)

        result_keys = f"{baseline_checkpoint}<->{checkpoint}" if baseline_checkpoint is not None else f"{baseline_checkpoint}<->{baseline_checkpoint}"

        result[result_keys] = {
            "win_rate_all": win_rate,
            "pivot": pivot_lang if pivot_lang is not None else None,
            "win_rate_from_pivot": win_rate_from_pivot if win_rate_from_pivot is not None else None,
            "win_rate_to_pivot": win_rate_to_pivot if win_rate_to_pivot is not None else None,
            "max_bleu_gain": max_bleu_gain,
            "max_bleu_gain_from_pivot": max_bleu_gain_from_pivot,
            "max_bleu_gain_to_pivot": max_bleu_gain_to_pivot,
            "min_bleu_gain": min_bleu_gain,
            "min_bleu_gain_from_pivot": min_bleu_gain_from_pivot,
            "min_bleu_gain_to_pivot": min_bleu_gain_to_pivot,
        }
    
    save_json(result, args.result_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="json file path of baseline")
    parser.add_argument("--others", required=True)
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--validate_checkpoints", default=False, action="store_true", help="validate checkpoint when nessasery")
    parser.add_argument("--pivot_lang")
    args = parser.parse_args()
    _get_win_rate(args)


if __name__ == "__main__":
    main()
