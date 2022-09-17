import argparse

_langdetect_installed = True
_langid_installed = True

try:
    import langdetect
    from langdetect import detect
    # achieve deterministic results
    langdetect.DetectorFactory.seed = 0
except ImportError:
    print("langdetect package has not installed yet!")
    _langdetect_installed = False

try:
    import langid
except ImportError:
    print("langid package has not installed yet!")
    _langid_installed = False


from collections import Counter
from nmt.data_handling import read_data


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--lang_code", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--detect_method", choices=["langdetect", "langid"], default="langdetect")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.detect_method == "langdetect":
        assert _langdetect_installed
    elif args.detect_method == "langid":
        assert _langid_installed

    acc_dict = Counter()
    
    unknown = 0
    inputs = read_data(args.inputs)
    for line in inputs:
        if args.detect_method == "langdetect":
            try:
                detect_lang_code = detect(line)
            except langdetect.lang_detect_exception.LangDetectException:
                unknown += 1
            else:
                detect_lang_code = detect_lang_code.replace("-", "_")
                acc_dict[detect_lang_code] += 1
        else:
            detect_lang_code = langid.classify(line)[0]
            acc_dict[detect_lang_code] += 1
    
    # sort by frequency
    acc_dict = Counter({lang: freq for lang, freq in sorted(acc_dict.items(), key=lambda item: item[1], reverse=True)})

    if unknown > 0:
        acc_dict["unknown"] = unknown

    if args.lang_code == "zh" and args.detect_method == "langdetect":
        correct = acc_dict["zh_ch"] + acc_dict["zh_tw"]
    elif args.lang_code in {"zh_cn", "zh_tw"} and args.detect_method == "langid":
        correct = acc_dict["zh"]
    else:
        correct = acc_dict[args.lang_code]
    
    print("{} acc is: {}".format(args.inputs, correct / len(inputs)))
    
    if args.verbose:
        for lang, freq in acc_dict.items():
            print("{:<10}{}".format(lang, freq))


if __name__ == "__main__":
    main()
