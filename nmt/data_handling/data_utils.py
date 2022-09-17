import random
import nltk
import re
import unicodedata
import os
import json

from typing import List, Optional, Any

from nmt.tools import Converter


def read_data(data_path: str, remove_chars: Optional[str]=None):
    with open(data_path, encoding="utf-8") as f:
        data = [line.strip(remove_chars) for line in f]
        return data


def write_data(data: List[str], data_path: str):
    data_path = os.path.abspath(data_path)
    directory = os.path.dirname(data_path)

    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    data = ["{}\n".format(line) for line in data]
    with open(data_path, "w", encoding="utf-8") as f:
        f.writelines(data)


def shuffle_corpus(src_data: List[str], tgt_data: List[str]):
    if tgt_data is None:
        random.shuffle(src_data)
    else:
        assert len(src_data) == len(tgt_data)
        if len(src_data) == 0:
            src_data, tgt_data = [], []
        else:
            all_data = list(zip(src_data, tgt_data))
            random.shuffle(all_data)
            src_data, tgt_data = list(zip(*all_data))
    return src_data, tgt_data


def remove_duplicate_sentence(src_data: List[str], tgt_data: List[str]):
    if tgt_data is None:
        src_data = list(set(src_data))
    else:
        assert len(src_data) == len(tgt_data)
        visited_data = set()
        src_data_new = []
        tgt_data_new = []

        for src_line, tgt_line in zip(src_data, tgt_data):
            if (src_line, tgt_line) not in visited_data:
                src_data_new.append(src_line)
                tgt_data_new.append(tgt_line)
                visited_data.add((src_line, tgt_line))
        
        src_data = src_data_new
        tgt_data = tgt_data_new
    return src_data, tgt_data


def remove_empty_line(src_data: List[str], tgt_data: List[str]):
    if tgt_data is None:
        src_data = list(filter(lambda item: len(item.strip()) > 0, src_data))
    else:
        assert len(src_data) == len(tgt_data)
        if len(src_data) == 0:
            src_data, tgt_data = [], []
        else:
            all_data = list(zip(src_data, tgt_data))
            all_data = list(filter(lambda item: len(item[0].strip()) > 0 and len(item[1].strip()) > 0, all_data))
            src_data, tgt_data = list(zip(*all_data))
    return src_data, tgt_data


def cht_to_chs(line: str):
    line = Converter("zh-hans").convert(line)
    line.encode("utf-8")
    return line


def chs_to_cht(line: str):
    line = Converter("zh-hant").convert(line)
    line.encode("utf-8")
    return line


def cjk_deseg(text: str):
    """ Desegment function for Chinese, Japanese and Korean.

    Args:
        text: A string.

    Returns:
        The desegmented string.
    """

    def _strip(matched):
        return matched.group(1).strip()

    CHAR_SPACE_PATTERN1 = r"([\u2E80-\u9FFF\uA000-\uA4FF\uAC00-\uD7FF\uF900-\uFAFF]\s+)"
    CHAR_SPACE_PATTERN2 = r"(\s+[\u2E80-\u9FFF\uA000-\uA4FF\uAC00-\uD7FF\uF900-\uFAFF])"

    res = re.sub(CHAR_SPACE_PATTERN1, _strip, text)
    res = re.sub(CHAR_SPACE_PATTERN2, _strip, res)
    
    res = re.sub(r',', r'，', res)
    res = re.sub(r'\?', r'？', res)
    res = re.sub(r'!', r'！', res)

    # no leading space
    res = re.sub(r'^\s+', r'', res)

    # no trailing space
    res = re.sub(r'\s+$', r'', res)
    return res


def sent_tokenize(text: str):
    pattern = re.compile(r"[\r]")
    text = pattern.sub(r"\n", text)
    
    sentences = []
    for sentence in text.split("\n"):
        sentences.append(nltk.tokenize.sent_tokenize(sentence))

    return sentences


def remove_long_sentence(src_data: List[str], tgt_data: List[str], max_sentence_length: int):
    if tgt_data is None:
        src_data = list(filter(lambda line: len(line.split()) <= max_sentence_length, src_data))
    else:
        assert len(src_data) == len(tgt_data)
        if len(src_data) == 0:
            src_data, tgt_data = [], []
        else:
            all_data = list(zip(src_data, tgt_data))
            all_data = list(filter(lambda item: max(len(item[0].split()), len(item[1].split())) <= max_sentence_length, all_data))
            if len(all_data) == 0:
                src_data, tgt_data = [], []
            else:
                src_data, tgt_data = list(zip(*all_data))
    return src_data, tgt_data


def is_chinese_char(ch):
    cp = ord(ch)
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def is_punctuation(ch):
    cp = ord(ch)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return True
    return False


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for ch in text:
        cp = ord(ch)
        if cp == 0 or cp == 0xfffd or is_control(ch):
            continue
        if is_whitespace(ch):
            output.append(" ")
        else:
            output.append(ch)
    return "".join(output)


def data_partition(data: Optional[List], parttion_num: int):
    """divide data to n partition"""
    if data is None:
        return None
    partition = []
    if len(data) < parttion_num:
        for i in range(parttion_num):
            partition.append([data[i]] if i < len(data) else [])
    else:
        block_size = len(data) // parttion_num

        batch = []
        for item in data:
            batch.append(item)
            if len(batch) == block_size and len(partition) < parttion_num:
                partition.append(batch)
                batch = []
        if len(batch) > 0:
            partition[-1].extend(batch)
    return partition


def drop_overlap_with_train(training_set_src: List[str], training_set_tgt: List[str], test_set_src: List[str]):
    """remove parallel sentences from training set that has duplicate source sentences with test set"""
    assert len(training_set_src) == len(training_set_tgt)

    test_set_src = set(test_set_src)
    training_set_src_new = []
    training_set_tgt_new = []

    for src, tgt in zip(training_set_src, training_set_tgt):
        if src not in test_set_src:
            training_set_src_new.append(src)
            training_set_tgt_new.append(tgt)
    
    return training_set_src_new, training_set_tgt_new


def load_json(data_path: str):
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, data_path: str):
    with open(data_path, mode="w", encoding="utf-8") as f:
        json.dump(data, f)


def normalize_line(line: str):
    normalize_line.normalizer = getattr(normalize_line, "normalizer", re.compile(r"\s+"))
    line = normalize_line.normalizer.sub(" ", line)
    line = line.strip()
    return line


def run_split_by_punc(text):
    """Split punctuation on a piece of text"""
    start_new_word = True
    output = []
    for char in text:
        if is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
    return " ".join("".join(x) for x in output)
