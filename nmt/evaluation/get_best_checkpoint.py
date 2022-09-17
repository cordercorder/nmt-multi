import argparse

from argparse import Namespace

from nmt.data_handling import load_json


def get_best_checkpoint(args: Namespace):
    valid_bleu_statistic = load_json(args.valid_bleu_statistic)

    num_lang_pair = len(valid_bleu_statistic[next(iter(valid_bleu_statistic))])

    result = {}

    for checkpoint in valid_bleu_statistic.keys():
        assert len(valid_bleu_statistic[checkpoint]) == num_lang_pair
        sum_bleu = 0.0
        for lang_pair, bleu_point in valid_bleu_statistic[checkpoint].items():
            sum_bleu += bleu_point
        
        result[checkpoint] = {
            "sum_bleu": sum_bleu,
            "avg_bleu": sum_bleu / num_lang_pair
        }

    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1]["sum_bleu"], reverse=True)}
    print(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_bleu_statistic", required=True, help="json data")
    args = parser.parse_args()
    get_best_checkpoint(args)


if __name__ == "__main__":
    main()
