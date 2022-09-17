import re
import argparse


from nmt.data_handling import read_data, save_json


PATTERN = re.compile(r".*?(?P<src_lang>\w*?)-(?P<tgt_lang>\w*?)\.\2 acc is: (?P<langacc>.*)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="json path")
    args = parser.parse_args()

    input = read_data(args.input)
    result = {}

    for line in input:
        if len(line) == 0:
            continue
        match_obj = PATTERN.fullmatch(line)
        if match_obj is None:
            continue
        src_lang = match_obj.group("src_lang")
        tgt_lang = match_obj.group("tgt_lang")
        langacc = float(match_obj.group("langacc"))
        key = f"{src_lang}-{tgt_lang}.{tgt_lang}"
        assert key not in result, f"{result[key]}"
        result[key] = langacc
    
    MEAN_LANG_ACC = sum(result.values()) / len(result)
    result["MEAN_LANG_ACC"] = MEAN_LANG_ACC
    save_json(result, args.output)


if __name__ == "__main__":
    main()
