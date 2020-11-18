# -*- coding: utf-8 -*-
import torch
import argparse
import pickle
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
import random
import os
import json


def _convert_to_transformer_inputs(fact, tokenizer, max_sequence_length, truncation_strategy='longest_first'):
    """
    Converts tokenized input to ids, masks and segments for transformer (including bert)
    :arg
     - fact: 一条事实描述
     - tokenizer: 分词器,可以把句子分成词,并把词映射到词表里的id,并且还可以给2个句子拼接成一条数据, 用[SEP]隔开2个句子
     - max_sequence_length: 拼接后句子的最长长度
     - truncation_strategy: 如果句子超过max_sequence_length,截断的策略

    :return
     - input_ids: 记录句子里每个词对应在词表里的 id
     - segments: 列表用来指定哪个是第一个句子，哪个是第二个句子，0的部分代表句子一, 1的部分代表句子二
     - input_masks: 列表中， 1的部分代表句子单词，而后面0的部分代表paddig，只是用于保持输入整齐，没有实际意义。
       相当于告诉BertModel不要利用后面0的部分
    """

    inputs = tokenizer.encode_plus(fact,
                                   text_pair=None,
                                   add_special_tokens=True,
                                   max_length=max_sequence_length,
                                   truncation_strategy=truncation_strategy,
                                   # truncation=True
                                   )

    input_ids = inputs["input_ids"]
    input_masks = [1] * len(input_ids)
    input_segments = inputs["token_type_ids"]
    padding_length = max_sequence_length - len(input_ids)
    padding_id = tokenizer.pad_token_id
    if padding_length > 0:
        # 长度不够，需要补齐
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

    return input_ids, input_masks, input_segments


def count_imprisonment(term_of_imprisonment):
    """
    :param term_of_imprisonment: dict形式
    :return:罪刑月数,有0~302类
    """
    if term_of_imprisonment["death_penalty"]:
        return 301
    elif term_of_imprisonment["death_penalty"]:
        return 302
    else:
        return term_of_imprisonment["imprisonment"]


def get_inoutput(dicList, tokenizer, max_sequence_length):
    """
    :param
     - dicList: 数据集集的字典 的 列表
     - tokenizer: 分词器
     - max_sequence_length: 拼接后句子的最长长度
    :return:
        三个处理好的tensor,形状都是[数据总数,max_sequence_length],它们的含义请看_convert_to_transformer_inputs
     - tokens_tensor: (tensor) [数据总数,max_sequence_length]
     - segments_tensors : (tensor) [数据总数,max_sequence_length]
     - input_masks_tensors: (tensor) [数据总数,max_sequence_length]
    """
    token_ids, masks, segments = [], [], []
    labels = []
    # 每一条数据
    for i in tqdm(range(len(dicList))):
        dic = dicList[i]
        term_of_imprisonment = dic["meta"]["term_of_imprisonment"]
        labels.append(count_imprisonment(term_of_imprisonment))
        input_ids, input_masks, input_segments = _convert_to_transformer_inputs(dic["fact"], tokenizer,
                                                                                max_sequence_length)
        token_ids.append(input_ids)
        masks.append(input_masks)
        segments.append(input_segments)

    tokens_tensor = torch.tensor(token_ids)
    segments_tensors = torch.tensor(segments)
    input_masks_tensors = torch.tensor(masks)

    return [tokens_tensor, segments_tensors, input_masks_tensors], torch.tensor(labels)


def clean_up(lines, sp):
    max_len = 0
    new_lines = []
    for line in lines:
        dic = json.loads(line)
        term_of_imprisonment = dic["meta"]["term_of_imprisonment"]["imprisonment"]
        # 有期徒刑最多 25 年，即300个月
        if term_of_imprisonment > 300:
            dic["meta"]["term_of_imprisonment"]["imprisonment"] = 300
        # 去掉文书里的空白字符
        # pattern = re.compile("\".*\"")
        # searchObj = re.search(pattern, dic["fact"])
        # if searchObj:
        #     print(searchObj.group())
        dic["fact"] = dic["fact"].replace("\r", "")
        dic["fact"] = dic["fact"].replace("\"", "'")
        dic["fact"] = dic["fact"].replace("\n", "")
        dic["fact"] = dic["fact"].replace("\\", "")
        new_len = len(dic["fact"])
        # 新
        new_line = json.dumps(dic)
        new_line = new_line.encode('utf-8').decode('unicode_escape')
        if new_len > max_len:
            max_len = new_len
            # print(dic["fact"])
        new_lines.append(new_line + "\n")
    print("{} dataset has max lenth: {}".format(sp, max_len))
    path = os.path.join(args.data_dir, "new_{}.json".format(sp))
    fout = open(path, "w")
    fout.writelines(new_lines)


def dump_to_pkl(sp):
    print("save {} dataset to pkl".format(sp))
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_dir)
    # train 从json
    if sp == "train":
        fin = open(os.path.join(args.data_dir, "new_train.json"), "r")
    else:
        fin = open(os.path.join(args.data_dir, "new_test.json"), "r")

    lines = fin.readlines()
    random.shuffle(lines)
    if sp == "train":
        # 截取到50000条
        if len(lines) > 50000:
            lines = lines[:50000]
    else:
        if len(lines) > 10000:
            lines = lines[:10000]
    # try:
    datas = [json.loads(line, strict=False) for line in lines]
    inputs, outputs = get_inoutput(datas, tokenizer, args.max_input_len)
    # 存到pkl里
    if sp == "train":
        with open(args.train_data_dir, 'wb') as fout:
            pickle.dump(inputs, fout)
            pickle.dump(outputs, fout)
    else:
        with open(args.test_data_dir, 'wb') as fout:
            pickle.dump(inputs, fout)
            pickle.dump(outputs, fout)


def work(args):
    # train 和 test 从json
    if args.clean_up:
        print("[0] clean up the blank str and modify imprison time length")
        train_fin = open(args.js_train_path, "r")
        lines = train_fin.readlines()
        clean_up(lines, "train")

        test_fin = open(args.js_test_path, "r")
        lines = test_fin.readlines()
        clean_up(lines, "test")
    print("[1] convert json to tensor, then save it at {}".format(args.data_dir))
    dump_to_pkl("train")
    print("[2] finish preprocess on  train set")
    dump_to_pkl("test")
    print("[3] finish preprocess on  test set")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--save-dir', type=str, default='./ckpt')
    parser.add_argument('--pretrained-dir', type=str, default='./albert_chinese_base/')
    # process parameters
    parser.add_argument('--max-input-len', type=int, default=1024)
    parser.add_argument('--clean_up', action='store_true')

    args = parser.parse_args()
    args.js_train_path = os.path.join(args.data_dir, "train.json")
    args.js_test_path = os.path.join(args.data_dir, "test.json")
    args.train_data_dir = os.path.join(args.data_dir, "train.pkl")
    args.test_data_dir = os.path.join(args.data_dir, "test.pkl")
    work(args)
