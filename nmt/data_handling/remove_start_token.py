import argparse

from nmt.data_handling import read_data, write_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_corpus", required=True)
    parser.add_argument("--tgt_corpus", required=True)
    parser.add_argument("--remove_token_src", action="store_true")
    parser.add_argument("--remove_token_tgt", action="store_true")

    parser.add_argument("--output_src_corpus", required=True)
    parser.add_argument("--output_tgt_corpus", required=True)

    args = parser.parse_args()
    assert args.remove_token_src or args.remove_token_tgt

    src_data = read_data(args.src_corpus)
    tgt_data = read_data(args.tgt_corpus)

    if args.remove_token_src:
        src_data = [line[line.find(" ")+1:] for line in src_data]
    if args.remove_token_tgt:
        tgt_data = [line[line.find(" ")+1:] for line in tgt_data]
    
    write_data(src_data, args.output_src_corpus)
    write_data(tgt_data, args.output_tgt_corpus)


if __name__ == "__main__":
    main()
