import argparse


from argparse import Namespace
from nmt.data_handling import read_data, write_data, normalize_line


def _main_(args: Namespace):
    paralle_table = {}
    src_data_a = [normalize_line(line) for line in read_data(args.src_corpus_a)]
    tgt_data_a = read_data(args.tgt_corpus_a)

    src_data_b = [normalize_line(line) for line in read_data(args.src_corpus_b)]
    tgt_data_b = read_data(args.tgt_corpus_b)
    
    for src_line, tgt_line in zip(src_data_a, tgt_data_a):
        paralle_table.setdefault(src_line, []).append(tgt_line)
    
    result_src_data = []
    result_tgt_data = []
    for src_line, tgt_line in zip(src_data_b, tgt_data_b):
        if src_line in paralle_table:
            for tgt_line_a in paralle_table[src_line]:
                result_src_data.append(tgt_line_a)
                result_tgt_data.append(tgt_line)
    
    write_data(result_src_data, args.output_src_data_path)
    write_data(result_tgt_data, args.output_tgt_data_path)


def main():
    # compare src_corpus_a and src_corpus_b to obtain parallel corpus of tgt_corpus_a -> tgt_corpus_b
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_corpus_a", required=True)
    parser.add_argument("--tgt_corpus_a", required=True)

    parser.add_argument("--src_corpus_b", required=True)
    parser.add_argument("--tgt_corpus_b", required=True)

    parser.add_argument("--output_src_data_path", required=True)
    parser.add_argument("--output_tgt_data_path", required=True)

    args = parser.parse_args()
    _main_(args)


if __name__ == "__main__":
    main()
