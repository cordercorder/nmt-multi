import argparse
import pycountry

from argparse import Namespace

from nmt.data_handling import read_data, save_json


def create_langcode_table(args: Namespace):
    langcode_table = {}
    lang_dict = read_data(args.lang_dict)

    for langcode in lang_dict:
        alpha = {}
        if len(langcode) == 2:
            alpha["alpha_2"] = langcode
        elif len(langcode) == 3:
            alpha["alpha_3"] = langcode
        else:
            langcode_table[langcode] = {"ISO 639-1": "unknown", "ISO 639-3": "unknown", "name": "unknown"}
            continue
        result = pycountry.languages.get(**alpha)
        if result is None:
            langcode_table[langcode] = {"ISO 639-1": "unknown", "ISO 639-3": "unknown", "name": "unknown"}
        else:
            langcode_table[langcode] = {"ISO 639-1": result.alpha_2, "ISO 639-3": result.alpha_3, "name": result.name}
    
    save_json(langcode_table, args.langcode_table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_dict", required=True)
    parser.add_argument("--langcode_table", required=True, help="json data")
    args = parser.parse_args()
    create_langcode_table(args)


if __name__ == "__main__":
    main()
