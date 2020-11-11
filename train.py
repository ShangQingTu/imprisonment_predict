# coding=UTF-8
import torch
import torch.utils.data as Data
import argparse
from transformers import *
from tqdm import tqdm
from torch import nn, optim
import random
import numpy as np
import shutil
import os
import pandas as pd
import json
import logging
from sklearn.metrics import f1_score
from model import AlbertClassifierModel
from utils import setup_logger, MetricLogger, strip_prefix_if_present


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
    :return:罪刑年份,有0~27类
    """
    if term_of_imprisonment["death_penalty"]:
        return 26
    elif term_of_imprisonment["death_penalty"]:
        return 27
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


def get_output(df_train):
    """
    :param df_train: 训练集的dataFrame
    :return: (tensor) [num_vocab] 数据的标注,只有0和1,1代表这个reply回答了query
    """

    labels = df_train['label']
    return torch.tensor(labels)


def train(inputs, outputs, args, tokenizer, logger):
    """
     :param:
     - inputs: (list) 作为输入的tensor, 它是由get_input处理得的
     - outputs: (tensor) 作为标注的tensor, 它是由get_output处理得的
     - args: 一堆训练前规定好的参数
     - logger: 训练日志,可以把训练过程记录在./ckpt/log.txt
     :return: 训练结束
     """
    # 创建数据集
    torch_dataset = Data.TensorDataset(inputs[0], inputs[1], inputs[2], outputs)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=True)
    logger.info('[1] Building model')
    # 查看运行训练脚本时,所用的设备,如果有cuda,就用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 构造 model
    model = AlbertClassifierModel(num_topics=args.num_topics,
                                  out_channels=args.out_channels,
                                  max_input_len=args.max_input_len,
                                  kernel_size=args.kernel_size,
                                  dropout=args.dropout).to(device)
    model_kwargs = {k: getattr(args, k) for k in
                    {'num_topics', 'out_channels', 'max_input_len', 'kernel_size', 'dropout'}
                    }
    logger.info(model)
    # 优化器
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    meters = MetricLogger(delimiter="  ")
    # BCEWithLogitsLoss是不需要sigmoid的二分类损失函数
    criterion = nn.BCEWithLogitsLoss()
    # scheduler,在schedule_step的时候,把学习率乘0.1,目前只在第一个step做了这个下降
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.schedule_step], gamma=0.1)
    logger.info('[2] Start training......')
    for epoch_num in range(args.max_epoch):
        # example_num:一个epoch需要训练多少个batch
        example_num = outputs.shape[0] // args.batch_size
        for batch_iter, (input_ids, segments_tensor, attention_mask, label) in enumerate(loader):
            progress = epoch_num + batch_iter / example_num
            optimizer.zero_grad()
            # 正向传播
            pred = model(input_ids.to(device), segments_tensor.to(device), attention_mask.to(device))
            # 处理 label
            if label.shape[0] != args.batch_size:
                logger.info('last dummy batch')
                break
            label = label.view(args.batch_size, 1)
            label = label.to(device).float()
            loss = criterion(pred, label)

            # 反向传播
            loss.backward()
            optimizer.step()
            meters.update(loss=loss)
            # 每过0.01个epoch记录一次loss
            if (batch_iter + 1) % (example_num // 100) == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "progress: {prog:.2f}",
                            "{meters}",
                        ]
                    ).format(
                        prog=progress,
                        meters=str(meters),
                    )
                )
        # 验证这个epoch的效果
        score, thres = validate(tokenizer, model, device, args)
        logger.info("val")
        logger.info(score)
        logger.info("thres")
        logger.info(thres)
        save = {
            'kwargs': model_kwargs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        scheduler.step()
        # 每个epoch保留一个ckpt
        torch.save(save,
                   os.path.join(args.save_dir, 'model_epoch%d_val%.3f.pt' % (epoch_num, score)))


def validate(tokenizer, model, device, args):
    """
    :param
    - inputs: (list) 作为输入的tensor, 它是由get_input处理得的
    - outputs: (tensor) 作为标注的tensor, 它是由get_output处理得的
    - args: 一堆训练前规定好的参数
    :return:
    - f1: 最好的f1分数
    - t: 要得到最好的f1分数所需设置的threshold值,当模型预测二分类为1的概率prob>threshold时,我们就分类为1
    """
    test_fin = open(args.js_test_path, "r")
    lines = test_fin.readlines()
    test_datas = [json.loads(line) for line in lines]
    # 处理dict为tensor
    inputs, outputs = get_inoutput(test_datas, tokenizer, args.max_input_len)
    with torch.no_grad():
        torch_dataset = Data.TensorDataset(inputs[0], inputs[1], inputs[2], outputs)
        # radom choose
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=True)
        pred_probs = []
        labels = []
        for batch_iter, (input_ids, segments_tensor, attention_mask, label) in enumerate(loader):
            # 测2000条来验证这个epoch的效果
            if batch_iter * args.batch_size > 2000:
                break
            pred_prob = model(input_ids.to(device), segments_tensor.to(device), attention_mask.to(device))
            pred_probs.append(pred_prob)
            labels.append(label.int())
        df_probs = pd.DataFrame(torch.cat(pred_probs).view(-1).cpu().numpy())
        df_labels = pd.DataFrame(torch.cat(labels).view(-1).cpu().numpy())
        f1, t = search_best_f1(df_probs, df_labels)
    return f1, t


def search_best_f1(pred_y, true_y):
    """
      :param
      - pred_y: (dataFrame) 模型对若干条数据,预测的二分类为1的概率组成的列表
      - true_y: (dataFrame) 这若干条数据实际的二分类标注
      :return:
      - best_f1_score: 最好的f1分数
      - best_thres: 要得到最好的f1分数所需设置的threshold值,当模型预测二分类为1的概率prob>threshold时,我们就分类为1
      """
    best_f1_score = 0
    best_thres = 0
    for i in range(-70, 70):
        tres = i / 100
        y_pred_bin = (pred_y > tres).astype(int)
        score = f1_score(true_y, y_pred_bin)
        if score > best_f1_score:
            best_f1_score = score
            best_thres = tres
    print('best_f1_score', best_f1_score)
    print('best_thres', best_thres)
    return best_f1_score, best_thres


def work(args):
    """
    :param args: 一堆训练前规定好的参数
    :return: 训练结束
    """
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_dir)
    # train 从json
    train_fin = open(args.js_train_path, "r")
    lines = train_fin.readlines()
    train_datas = [json.loads(line) for line in lines]
    # 处理dict为tensor
    train_inputs, train_outputs = get_inoutput(train_datas, tokenizer, args.max_input_len)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    logger = setup_logger("Classify", args.save_dir)

    # args display
    for k, v in vars(args).items():
        logger.info(k + ':' + str(v))

    train(train_inputs, train_outputs, args, tokenizer, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--data-dir', type=str, default='./data/processed')
    parser.add_argument('--save-dir', type=str, default='./ckpt')
    parser.add_argument('--pretrained-dir', type=str, default='./albert_chinese_base/')
    # model parameters, see them in `model.py`
    parser.add_argument('--num-topics', type=int, default=1)
    parser.add_argument('--max-input-len', type=int, default=1024)
    parser.add_argument('--out-channels', type=int, default=2)
    parser.add_argument('--kernel-size', type=int, nargs='+', default=[2, 3, 4])

    # training parameters
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--schedule_step', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay rate per batch')
    parser.add_argument('--seed', type=int, default=666666, help='random seed')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--optim', default='adam', choices=['adam', 'adamw', 'sgd'])
    args = parser.parse_args()
    args.js_train_path = os.path.join(args.data_dir, "new_train.json")
    args.js_test_path = os.path.join(args.data_dir, "new_test.json")
    work(args)
