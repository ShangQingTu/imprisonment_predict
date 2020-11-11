# coding=UTF-8
import torch
import torch.utils.data as Data
import argparse
from transformers import *
from tqdm import tqdm
import os
import pandas as pd
import re
from pandas.core.frame import DataFrame
from model import AlbertClassifierModel
from utils import strip_prefix_if_present
from train import get_input, get_output


def work(args):
    """
    :param args: 一堆 运行前 规定好的 参数
    :return: 在测试集上输出结束(会输出到args.output_dir),
    如果用了规则(--use_rule),那么会有规则和模型预测prob冲突的记录
    记录在args.output_dir/scores.txt
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载test数据
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_dir)
    df_test = pd.read_csv(args.df_test_path, header=None)
    df_test.columns = ['id', 'id_sub', 'query', 'reply']
    inputs = get_input(df_test, tokenizer, args.max_input_len)
    # 加载ckpt
    loaded = torch.load(args.ckpt)
    model_kwargs = loaded['kwargs']
    for k in model_kwargs:
        # 如果现有的args里有和训练时冲突的,就把加载进来的ckpt的参数覆盖掉
        if hasattr(args, k) and getattr(args, k) is not None:
            model_kwargs[k] = getattr(args, k)
    args.fout.write("state_dict raw")
    print(loaded['state_dict'])
    # TODO 是否要strip_prefix
    # loaded['state_dict'] = strip_prefix_if_present(loaded['state_dict'], prefix='module.')
    for k, v in model_kwargs.items():
        args.fout.write("k {}, v{}".format(k, v))
        print(k, v)

    # 加载模型
    model = AlbertClassifierModel(**model_kwargs)
    model.load_state_dict(loaded['state_dict'])
    model.eval()
    model = model.to(device)
    # 预测
    with torch.no_grad():
        predict_on_testset(model, inputs, df_test, device, args)


def predict_on_testset(model, inputs, df_test, device, args):
    """
    :param model: 模型对象
    :param inputs: (list) 作为输入的tensor, 它是由get_input处理得的
    :param df_test: test集原始数据的 dataFrame
    :param device: cuda 或 cpu
    :param args:一堆 运行前 规定好的 参数
    :return: 测试集输出结束
    """
    torch_dataset = Data.TensorDataset(inputs[0], inputs[1], inputs[2])
    # 这是test集,不要打乱,故shuffle=False
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=False)
    pred_y_list = []
    pred_prob_list = []
    for batch_iter, (input_ids, segments_tensor, attention_mask) in tqdm(enumerate(loader)):
        pred_prob_tensor = model(input_ids.to(device), segments_tensor.to(device), attention_mask.to(device))
        pred_prob_tensor = pred_prob_tensor.view(-1)
        prob = pred_prob_tensor.cpu().numpy()
        if prob > args.thres:
            pred_y_list.append(1)
        else:
            pred_y_list.append(0)
        pred_prob_list.append(prob)

    assert len(pred_y_list) == len(df_test)
    # 拼接df, 保存为csv
    pred_dic = {"pred": pred_y_list, "pred_prob": pred_prob_list}
    df_pred = DataFrame(pred_dic)
    try:
        df_res = pd.concat([df_test, df_pred], axis=1).reset_index(drop=True)
        df_final = check_rule(df_res, args)
        df_final[['id', 'id_sub', 'pred']].to_csv(args.output_csv, index=False, header=None,
                                                  sep='\t')
    except Exception as  e:
        print(e)
    df_pred.to_csv(os.path.join(args.output_dir, "pred.csv"))
    df_test.to_csv(os.path.join(args.output_dir, "test.csv"), index=False, header=None)


def check_rule(df_res, args):
    """
    :param df_res: 模型预测结果的dataFrame
    :param args: 一堆 运行前 规定好的 参数
    :return: 规则应用完了
    """
    if args.use_rule:
        for i in range(len(df_res)):
            query = df_res.iloc[i]['query']
            reply = df_res.iloc[i]['reply']
            for key, pattern in args.str2pattern_query.items():
                searchObj = re.search(pattern, query)
                if searchObj:
                    if df_res.iloc[i]["pred"] == 1:
                        prob = df_res.iloc[i]["pred_prob"]
                        args.fout.write(
                            "at line {}, query {}, reply {},rule {} effcts, model pred prob:{}".format(i, query, reply,
                                                                                                       key, prob))
                        df_res.iloc[i]["pred"] = 0

            for key, pattern in args.str2pattern_reply.items():
                searchObj = re.search(pattern, query)
                if searchObj:
                    if df_res.iloc[i]["pred"] == 1:
                        prob = df_res.iloc[i]["pred_prob"]
                        args.fout.write(
                            "at line {}, query {}, reply {},rule {} effcts, model pred prob:{}".format(i, query, reply,
                                                                                                       key, prob))
                        df_res.iloc[i]["pred"] = 0

    return df_res


def initPatternDic(args):
    """
    :param args: 一堆 运行前 规定好的 参数
    :return: 初始化正则表达式字典
    """
    args.str2pattern_query = {}
    args.str2pattern_reply = {}
    # key: 关键字 value:正则pattern
    pattern1 = re.compile(".*我正在看.*")
    args.str2pattern_query["我正在看"] = pattern1
    pattern1 = re.compile(".*我想咨询.*")
    args.str2pattern_query["我想咨询"] = pattern1
    pattern1 = re.compile(".*PHONE.*")
    args.str2pattern_reply["PHONE"] = pattern1
    pattern1 = re.compile(".*NAME.*")
    args.str2pattern_reply["NAME"] = pattern1
    # pattern1 = re.compile(".*小学.*")
    # args.str2pattern["小学"] = pattern1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--data-dir', type=str, default='./data/processed')
    parser.add_argument('--save-dir', type=str, default='./ckpt')
    parser.add_argument('--pretrained-dir', type=str, default='./albert_chinese_base/')
    parser.add_argument('--ckpt', required=True)
    # test parameters
    parser.add_argument('--thres', type=float, default=0.15)
    parser.add_argument('--use_rule', action="store_true")

    # model parameters
    parser.add_argument('--max-input-len', type=int, default=100)
    # mini-batch 保证每一条test都有输出
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()
    initPatternDic(args)
    # test集的路径
    args.df_test_path = os.path.join(args.data_dir, "test.combine_beike.csv")

    # write the output [id,sub_id,pred_y]
    ckpt_base = os.path.splitext(os.path.basename(args.ckpt))[0]

    output_dir = os.path.join(args.save_dir, "test_output_{}".format(ckpt_base))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    else:
        print('output dir %s exists, be careful that we will overwrite it' % output_dir)
    # 最终结果
    args.output_dir = output_dir
    args.output_csv = os.path.join(output_dir, 'result.csv')
    f = open(os.path.join(output_dir, 'scores.txt'), 'w')
    args.fout = f
    work(args)
