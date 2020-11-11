# -*- coding: utf-8 -*-
import re
import argparse
import random
import numpy as np
import shutil
import os
import pandas as pd
import json


def clean_up(lines, sp):
    max_len = 0
    new_lines = []
    for line in lines:
        dic = json.loads(line)
        term_of_imprisonment = dic["meta"]["term_of_imprisonment"]["imprisonment"]
        # 有期徒刑最多 25
        if term_of_imprisonment > 25:
            dic["meta"]["term_of_imprisonment"]["imprisonment"] = 25
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
    print("{} dataset has max lenth: {}".format(sp,max_len))
    path = os.path.join(args.data_dir, "new_{}.json".format(sp))
    fout = open(path, "w")
    fout.writelines(new_lines)


def work(args):
    # train 和 test 从json
    train_fin = open(args.js_train_path, "r")
    lines = train_fin.readlines()
    clean_up(lines, "train")

    test_fin = open(args.js_test_path, "r")
    lines = test_fin.readlines()
    clean_up(lines, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--save-dir', type=str, default='./ckpt')
    parser.add_argument('--pretrained-dir', type=str, default='./albert_chinese_base/')

    args = parser.parse_args()
    args.js_train_path = os.path.join(args.data_dir, "train.json")
    args.js_test_path = os.path.join(args.data_dir, "test.json")
    work(args)
