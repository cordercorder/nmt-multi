import argparse
import random

from argparse import Namespace

from nmt.data_handling import (
    read_data, 
    write_data, 
    shuffle_corpus, 
    remove_duplicate_sentence, 
    remove_long_sentence, 
    remove_empty_line
)

def validate_args(args: Namespace):
    if args.operation == "remove_long_sentence":
        assert args.max_sentence_length is not None


def _main(args: Namespace):
    validate_args(args)
    random.seed(args.seed)

    src_data = read_data(args.src_path)
    tgt_data = read_data(args.tgt_path)
    assert len(src_data) == len(tgt_data)

    src_data, tgt_data = remove_empty_line(src_data, tgt_data)

    if args.operation == "remove_same_sentence":
        src_data, tgt_data = remove_duplicate_sentence(src_data, tgt_data)
    elif args.operation == "remove_long_sentence":
        src_data, tgt_data = remove_long_sentence(src_data, tgt_data, args.max_sentence_length)
    else:
        src_data, tgt_data = shuffle_corpus(src_data, tgt_data)
    
    write_data(src_data, args.output_src_path)
    write_data(tgt_data, args.output_tgt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--src_path", required=True)
    parser.add_argument("--tgt_path", required=True)
    parser.add_argument("--output_src_path", required=True)
    parser.add_argument("--output_tgt_path", required=True)
    parser.add_argument("--operation", required=True, choices=["remove_same_sentence", "shuffle_corpus", "remove_long_sentence"])

    parser.add_argument("--max_sentence_length", type=int)

    args = parser.parse_args()
    _main(args)


if __name__ == "__main__":
    main()
    