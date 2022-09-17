import argparse

from fairseq.data import Dictionary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_files", required=True, nargs="+")
    parser.add_argument("--merged_dict", required=True)

    parser.add_argument("--finalize", action="store_true", default=False)
    parser.add_argument("--threshold", type=int, default=0, help="defines the minimum word count")
    parser.add_argument("--nwords", type=int, default=-1, help="defines the total number of words in the final dictionary, including special symbols")
    parser.add_argument("--padding_factor", type=int, default=8, help="can be used to pad the dictionary size to be a multiple of 8, which is important on some hardware (e.g., Nvidia Tensor Cores).")

    args = parser.parse_args()

    d = Dictionary.load(args.dict_files[0])

    for dict_file in args.dict_files[1:]:
        d.update(Dictionary.load(dict_file))
    
    if args.finalize:
        d.finalize(args.threshold, args.nwords, args.padding_factor)

    d.save(args.merged_dict)


if __name__ == "__main__":
    main()
