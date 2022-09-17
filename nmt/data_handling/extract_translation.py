import argparse


from typing import List, Tuple

from nmt.data_handling import read_data, write_data, cjk_deseg


def valid_tgt_data(tgt_data: List[Tuple[int, str]]):
    if len(tgt_data) == 0:
        print("Empty tgt data")
        return
    assert tgt_data[0][0] == 0
    for i in range(1, len(tgt_data)):
        if tgt_data[i][0] != tgt_data[i-1][0] + 1:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--translation_file_path", required=True)
    parser.add_argument("--output_tgt_file_path", required=True)
    parser.add_argument("--desegment_zh", action="store_true")

    args = parser.parse_args()

    translation_data = read_data(args.translation_file_path)
    tgt_data = []
    
    for line in translation_data:
        if line.startswith("D-"):
            line = line[2:].split("\t")
            assert 2 <= len(line) <= 3
            line_num = int(line[0])
            tgt = line[2] if len(line) == 3 else ""
            tgt_data.append((line_num, tgt))
    
    tgt_data.sort(key=lambda item: item[0])
    if valid_tgt_data(tgt_data):
        tgt_data = [item[1] for item in tgt_data]
        if args.desegment_zh:
            tgt_data = [cjk_deseg(line) for line in tgt_data]
        
        write_data(tgt_data, args.output_tgt_file_path)
    else:
        write_data([""], args.output_tgt_file_path)
        print("Error in translation file")


if __name__ == "__main__":
    main()
