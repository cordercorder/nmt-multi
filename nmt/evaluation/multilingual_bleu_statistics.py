import re
import argparse

from argparse import Namespace
from typing import Tuple
from functools import cmp_to_key


from nmt.data_handling import read_data, save_json


def extract_bleu(args: Namespace):
    def cmp(item_a: Tuple, item_b: Tuple):
        def get_result(item: Tuple):
            checkpoint = item[0]
            result = re.findall(r"\d+", checkpoint)
            assert 1 <= len(result) <= 2, f"item is: {item}"
            return result
        
        result_a = get_result(item_a)
        result_b = get_result(item_b)

        if len(result_a) == 1 and len(result_b) == 1:
            return int(result_a[0]) - int(result_b[0])
        elif len(result_a) == 1 and len(result_b) == 2:
            deta = int(result_a[0]) - int(result_b[0])
            if deta == 0:
                deta += 1
            return deta
        elif len(result_a) == 2 and len(result_b) == 1:
            deta = int(result_a[0]) - int(result_b[0])
            if deta == 0:
                deta -= 1
            return deta
        else:
            deta = int(result_a[0]) - int(result_b[0])
            if deta != 0:
                return deta
            deta = int(result_a[1]) - int(result_b[1])
            return deta

    data = read_data(args.input)

    # checkpoint30.pt.en-id.BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version = 25.288301 55.9/31.1/19.4/12.3 (BP = 0.996 ratio = 0.996 hyp_len = 16365 ref_len = 16431)
    pattern = re.compile(r"(?P<checkpoint>.*)\.pt\.(?P<src_lang>.*?)-(?P<tgt_lang>.*?)\.(.*?) = (?P<bleu_score>\d+.\d+)")
    
    result = {}
    best_loss_result = {}
 
    for line in data:
        match_obj = pattern.match(line)
        if match_obj:
            checkpoint = match_obj.group("checkpoint")
            src_lang = match_obj.group("src_lang")
            tgt_lang = match_obj.group("tgt_lang")

            bleu_score = float(match_obj.group("bleu_score"))
            lang_pair = "-".join([src_lang, tgt_lang])

            if "best_loss" in checkpoint or checkpoint == "checkpoint_best":
                best_loss_result.setdefault(checkpoint, {})
                assert lang_pair not in best_loss_result[checkpoint]
                best_loss_result[checkpoint][lang_pair] = bleu_score
            else:
                result.setdefault(checkpoint, {})
                assert lang_pair not in result[checkpoint]
                result[checkpoint][lang_pair] = bleu_score

    result = {checkpoint: lang_pair for checkpoint, lang_pair in sorted(result.items(), key=cmp_to_key(cmp))}
    best_loss_result = {checkpoint: lang_pair for checkpoint, lang_pair in sorted(best_loss_result.items(), key=lambda item: float(getattr(re.search(r"\d+.\d+", item[0]), "group", lambda: 0)()), reverse=True) }

    # merge two dict
    result = {**result, **best_loss_result}
    save_json(result, args.output_json_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_json_data", required=True)
    
    args = parser.parse_args()
    extract_bleu(args)


if __name__ == "__main__":
    main()
