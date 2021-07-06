import collections
import json
import os
from typing import Optional, List, Union, Dict
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

import tensorflow as tf
from transformers import BertTokenizer

from extract_chinese_and_punct import ChineseAndPunctuationExtractor

InputFeature = collections.namedtuple("InputFeature", [
    "input_ids", "seq_len", "tok_to_orig_start_index", "tok_to_orig_end_index",
    "labels"
])

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def parse_label(spo_list, label_map, tokens, tokenizer):
    # 2 tags for each predicate + I tag + O tag
    num_labels = 2 * (len(label_map.keys()) - 2) + 2
    seq_len = len(tokens)
    # initialize tag
    labels = [[0] * num_labels for i in range(seq_len)]
    #  find all entities and tag them with corresponding "B"/"I" labels
    for spo in spo_list:
        for spo_object in spo['object'].keys():
            # assign relation label
            if spo['predicate'] in label_map.keys():
                # simple relation
                label_subject = label_map[spo['predicate']]
                label_object = label_subject + 55
                subject_tokens = tokenizer._tokenize(spo['subject'])
                object_tokens = tokenizer._tokenize(spo['object']['@value'])
            else:
                # complex relation
                label_subject = label_map[spo['predicate'] + '_' + spo_object]
                label_object = label_subject + 55
                subject_tokens = tokenizer._tokenize(spo['subject'])
                object_tokens = tokenizer._tokenize(spo['object'][spo_object])

            subject_tokens_len = len(subject_tokens)
            object_tokens_len = len(object_tokens)

            # assign token label
            # there are situations where s entity and o entity might overlap, e.g. xyz established xyz corporation
            # to prevent single token from being labeled into two different entity
            # we tag the longer entity first, then match the shorter entity within the rest text
            forbidden_index = None
            if subject_tokens_len > object_tokens_len:
                for index in range(seq_len - subject_tokens_len + 1):
                    if tokens[index:index +
                                    subject_tokens_len] == subject_tokens:
                        labels[index][label_subject] = 1
                        for i in range(subject_tokens_len - 1):
                            labels[index + i + 1][1] = 1
                        forbidden_index = index
                        break

                for index in range(seq_len - object_tokens_len + 1):
                    if tokens[index:index + object_tokens_len] == object_tokens:
                        if forbidden_index is None:
                            labels[index][label_object] = 1
                            for i in range(object_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break
                        # check if labeled already
                        elif index < forbidden_index or index >= forbidden_index + len(
                                subject_tokens):
                            labels[index][label_object] = 1
                            for i in range(object_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break

            else:
                for index in range(seq_len - object_tokens_len + 1):
                    if tokens[index:index + object_tokens_len] == object_tokens:
                        labels[index][label_object] = 1
                        for i in range(object_tokens_len - 1):
                            labels[index + i + 1][1] = 1
                        forbidden_index = index
                        break

                for index in range(seq_len - subject_tokens_len + 1):
                    if tokens[index:index +
                                    subject_tokens_len] == subject_tokens:
                        if forbidden_index is None:
                            labels[index][label_subject] = 1
                            for i in range(subject_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break
                        elif index < forbidden_index or index >= forbidden_index + len(
                                object_tokens):
                            labels[index][label_subject] = 1
                            for i in range(subject_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break

    # if token wasn't assigned as any "B"/"I" tag, give it an "O" tag for outside
    for i in range(seq_len):
        if labels[i] == [0] * num_labels:
            labels[i][0] = 1

    return labels


def convert_example_to_feature(
        example,
        tokenizer: BertTokenizer,
        chineseandpunctuationextractor: ChineseAndPunctuationExtractor,
        label_map,
        max_length: Optional[int] = 512,
        pad_to_max_length: Optional[bool] = None):
    spo_list = example['spo_list'] if "spo_list" in example.keys() else None
    text_raw = example['text']

    sub_text = []
    buff = ""
    for char in text_raw:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
        else:
            buff += char
    if buff != "":
        sub_text.append(buff)

    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    orig_to_tok_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        orig_to_tok_index.append(len(tokens))
        sub_tokens = tokenizer._tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
            if len(tokens) >= max_length - 2:
                break
        else:
            continue
        break

    seq_len = len(tokens)
    # 2 tags for each predicate + I tag + O tag
    num_labels = 2 * (len(label_map.keys()) - 2) + 2
    # initialize tag
    labels = [[0] * num_labels for i in range(seq_len)]
    if spo_list is not None:
        labels = parse_label(spo_list, label_map, tokens, tokenizer)

    # add [CLS] and [SEP] token, they are tagged into "O" for outside
    if seq_len > max_length - 2:
        tokens = tokens[0:(max_length - 2)]
        labels = labels[0:(max_length - 2)]
        tok_to_orig_start_index = tok_to_orig_start_index[0:(max_length - 2)]
        tok_to_orig_end_index = tok_to_orig_end_index[0:(max_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # "O" tag for [PAD], [CLS], [SEP] token
    outside_label = [[1] + [0] * (num_labels - 1)]

    labels = outside_label + labels + outside_label
    tok_to_orig_start_index = [-1] + tok_to_orig_start_index + [-1]
    tok_to_orig_end_index = [-1] + tok_to_orig_end_index + [-1]
    if seq_len < max_length:
        tokens = tokens + ["[PAD]"] * (max_length - seq_len - 2)
        labels = labels + outside_label * (max_length - len(labels))
        tok_to_orig_start_index = tok_to_orig_start_index + [-1] * (
                max_length - len(tok_to_orig_start_index))
        tok_to_orig_end_index = tok_to_orig_end_index + [-1] * (
                max_length - len(tok_to_orig_end_index))

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return InputFeature(
        input_ids=np.array(token_ids),
        seq_len=np.array(seq_len),
        tok_to_orig_start_index=np.array(tok_to_orig_start_index),
        tok_to_orig_end_index=np.array(tok_to_orig_end_index),
        labels=np.array(labels))


class DuIEDataset():
    """
    Dataset of DuIE.
    """

    def __init__(
            self,
            input_ids: List[Union[List[int], np.ndarray]],
            seq_lens: List[Union[List[int], np.ndarray]],
            tok_to_orig_start_index: List[Union[List[int], np.ndarray]],
            tok_to_orig_end_index: List[Union[List[int], np.ndarray]],
            labels: List[Union[List[int], np.ndarray, List[str], List[Dict]]]):
        pass

    @classmethod
    def from_file(cls,
                  file_path: Union[str, os.PathLike],
                  tokenizer: BertTokenizer,
                  max_length: Optional[int] = 512,
                  pad_to_max_length: Optional[bool] = None):
        def gen():
            assert os.path.exists(file_path) and os.path.isfile(
                file_path), f"{file_path} dose not exists or is not a file."
            label_map_path = os.path.join(
                os.path.dirname(file_path), "predicate2id.json")
            assert os.path.exists(label_map_path) and os.path.isfile(
                label_map_path
            ), f"{label_map_path} dose not exists or is not a file."
            with open(label_map_path, 'r', encoding='utf8') as fp:
                label_map = json.load(fp)
            chineseandpunctuationextractor = ChineseAndPunctuationExtractor()

            # input_ids, seq_lens, tok_to_orig_start_index, tok_to_orig_end_index, labels = ([] for _ in range(5))
            # dataset_scale = sum(1 for line in open(file_path, 'r'))
            tf.compat.v1.logging.info("Preprocessing data, loaded from %s" % file_path)
            with open(file_path, "r", encoding="utf-8") as fp:
                while True:
                    line = fp.readline()
                    if not line:
                        break
                    example = json.loads(line)
                    input_feature = convert_example_to_feature(
                        example, tokenizer, chineseandpunctuationextractor,
                        label_map, max_length, pad_to_max_length)
                    # input_ids.append(input_feature.input_ids)
                    # seq_lens.append(input_feature.seq_len)
                    # tok_to_orig_start_index.append(
                    #     input_feature.tok_to_orig_start_index)
                    # tok_to_orig_end_index.append(
                    #     input_feature.tok_to_orig_end_index)
                    # labels.append(input_feature.labels)

                    yield input_feature.input_ids, input_feature.seq_len, input_feature.tok_to_orig_start_index, \
                          input_feature.tok_to_orig_end_index, input_feature.labels

        return gen
        # x, y = [], []
        # for item in zip(input_ids, seq_lens, tok_to_orig_start_index, tok_to_orig_end_index, labels):
        #     x.append(item[:-1])
        #     y.append(item[-1])
        #
        # return x, y


@dataclass
class DataCollator:
    """
    Collator for DuIE.
    """

    def __call__(self, examples: List[Dict[str, Union[list, np.ndarray]]]):
        batched_input_ids = np.stack([x['input_ids'] for x in examples])
        seq_lens = np.stack([x['seq_lens'] for x in examples])
        tok_to_orig_start_index = np.stack(
            [x['tok_to_orig_start_index'] for x in examples])
        tok_to_orig_end_index = np.stack(
            [x['tok_to_orig_end_index'] for x in examples])
        labels = np.stack([x['labels'] for x in examples])

        return (batched_input_ids, seq_lens, tok_to_orig_start_index,
                tok_to_orig_end_index, labels)


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    x, y = DuIEDataset.from_file("../data/dev_data_sample.json", tokenizer)

    for xx, yy in zip(x, y):
        model_input = {
            "input_ids": xx[0],
            "seq_len": xx[1],
            "tok_to_orig_start_index": xx[2],
            "tok_to_orig_end_index": xx[3],
            "labels": yy
        }
        print(model_input)
