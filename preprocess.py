import torch
import argparse
import os
import json
import pickle
import pandas as pd
import logging
import re

logging.basicConfig(level=logging.INFO)


def find_pattern_in_set(pattern, df, pattern_name, split="train", isQuery=True):
    """
    :param pattern: 正则表达式的 pattern
    :param df: 输入的数据集的 dataFrame
    :param pattern_name: pattern里的中文名,比如"我正在看"
    :param split: 表示是train集还是test集
    :param isQuery: True,、查 在问题文本里有pattern匹配的句子. False,查 在回答文本里有pattern匹配的句子
    :return: 打印在std的信息
    """
    print("{}集 combine_beike.csv".format(split))
    isLooking_qurey_num = 0
    isLooking_positive_num = 0
    for i in range(len(df)):
        query = df.iloc[i]['query']
        reply = df.iloc[i]['reply']
        if isQuery:
            searchObj = re.search(pattern, query)
        else:
            searchObj = re.search(pattern, reply)
        if searchObj:
            isLooking_qurey_num += 1
            if split == "train":
                label = df.iloc[i]['label']
                if label == 1:
                    isLooking_positive_num += 1
                    print("row {}".format(i))
                    print("label {}, qu: {} ".format(label, query))
                    print("label {}, re: {} ".format(label, reply))
    if isQuery:
        print("问题里含有 \"{}\" 的句子有{}个".format(pattern_name, isLooking_qurey_num))
        if split == "train":
            print("对 \"{}\"的回答标注为1的 有{}个".format(pattern_name, isLooking_positive_num))
            print("全部预测为0,accuracy: {} ".format(1 - isLooking_positive_num / isLooking_qurey_num))
    else:
        print("回答里含有 \"{}\" 的句子有{}个".format(pattern_name, isLooking_qurey_num))
        if split == "train":
            print("对含有 \"{}\"的回答标注为1的 有{}个".format(pattern_name, isLooking_positive_num))
            print("全部预测为0,accuracy: {} ".format(1 - isLooking_positive_num / isLooking_qurey_num))


def workTrain(args):
    """
    :param args: 一堆 运行前 规定好的 参数
    :return: 在 train集上查找的 pattern相关信息
    """
    train_left = pd.read_csv(args.train_query_dir, sep='\t', header=None)
    train_left.columns = ['id', 'query']
    train_right = pd.read_csv(args.train_reply_dir, sep='\t', header=None)
    train_right.columns = ['id', 'id_sub', 'reply', 'label']
    df_train = train_left.merge(train_right, how='left')
    # 补充空的值
    df_train['reply'] = df_train['reply'].fillna('可以')
    print('train shape =', df_train.shape)
    # 查找一些东西,观察数据的规律
    pattern1 = args.str2pattern[args.pattern_key]
    find_pattern_in_set(pattern1, df_train, args.pattern_key, "train", args.isQuery)
    if args.save_combine:
        df_train[['id', 'id_sub', 'query', 'reply', 'label']].to_csv(
            os.path.join(args.save_dir, 'train.combine_beike.csv'), index=False, header=None,
            sep=',')


def workTest(args):
    """
    :param args: 一堆 运行前 规定好的 参数
    :return: 在 test集上查找的 pattern相关信息
    """
    test_left = pd.read_csv(args.test_query_dir, sep='\t', header=None, encoding='gbk')
    test_left.columns = ['id', 'query']
    test_right = pd.read_csv(args.test_reply_dir, sep='\t', header=None, encoding='gbk')
    test_right.columns = ['id', 'id_sub', 'reply']
    df_test = test_left.merge(test_right, how='left')
    print('df_test shape =', df_test.shape)
    # 查找一些东西,观察数据的规律
    pattern1 = args.str2pattern[args.pattern_key]
    find_pattern_in_set(pattern1, df_test, args.pattern_key, "test", args.isQuery)
    if args.save_combine:
        df_test[['id', 'id_sub', 'query', 'reply']].to_csv(os.path.join(args.save_dir, "test.combine_beike.csv"),
                                                           index=False,
                                                           header=None,
                                                           sep=',')


def initPatternDic(args):
    """
    :param args: 一堆 运行前 规定好的 参数
    :return: 初始化正则表达式字典
    """
    args.str2pattern = {}
    # key: 关键字 value:正则pattern
    pattern1 = re.compile(".*我正在看.*")
    args.str2pattern["我正在看"] = pattern1
    pattern1 = re.compile(".*我想咨询.*")
    args.str2pattern["我想咨询"] = pattern1
    pattern1 = re.compile(".*PHONE.*")
    args.str2pattern["PHONE"] = pattern1
    pattern1 = re.compile(".*NAME.*")
    args.str2pattern["NAME"] = pattern1
    pattern1 = re.compile(".*小学.*")
    args.str2pattern["小学"] = pattern1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--save-dir', type=str, default='./data/processed')
    # 是否保存合并
    parser.add_argument('--save-combine', action='store_true')
    # pattern的key值
    parser.add_argument('--pattern-key', type=str, default='小学')
    # 查的pattern是问题里的
    parser.add_argument('--isQuery', action='store_true')
    # parser.add_argument('--pattern-key', type=str, default='我想咨询')

    args = parser.parse_args()

    # 训练数据
    args.train_dir = os.path.join(args.data_dir, "train")
    args.train_query_dir = os.path.join(args.train_dir, "train.query.tsv")
    args.train_reply_dir = os.path.join(args.train_dir, "train.reply.tsv")
    # test数据
    args.test_dir = os.path.join(args.data_dir, "test")
    args.test_query_dir = os.path.join(args.test_dir, "test.query.tsv")
    args.test_reply_dir = os.path.join(args.test_dir, "test.reply.tsv")
    # 初始化pattern字典
    initPatternDic(args)
    if not (os.path.exists(args.save_dir)):
        os.makedirs(args.save_dir)

    workTrain(args)
    # workTest(args)
