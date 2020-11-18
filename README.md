CAIL2018数据集在线实验

云服务器：滴滴云

## preprocess

- 先把原始数据里的空白字符去掉,并把25年(即25*12=300个月)以上的有期徒刑改成25年(法律是这么规定的)

- 再把json数据用Bert分词器处理为tensor，再存为pikcle方便读取

- 由于原始数据量太大，缩小到训练集50000条，测试集10000条

- 上传代码和数据到云服务器：

  - ```
    scp -r /home/tsq/PycharmProjects/learnCNN/imprisonment_predict  root@117.51.152.45:/root
    ```

    

- 预处理命令：

  - ```
    python preprocess.py --clean_up
    ```



## train

由于输入是很长的，设置了默认的最大输入长度为1024，多了会截取

- 训练命令:

  - ```
    python train.py
    ```

    

- 更新一下

  - ```
    scp /home/tsq/PycharmProjects/learnCNN/imprisonment_predict/*.py root@117.51.150.50:/root/imprisonment_predict
    ```

  - 报错

    ```bash
    2020-11-18 16:16:23,272 Classify INFO: [2] Start training......
    Traceback (most recent call last):
      File "train.py", line 205, in <module>
        work(args)
      File "train.py", line 176, in work
        train(train_inputs, train_outputs, args, logger)
      File "train.py", line 70, in train
        attention_mask.to(device).view(batch_size, -1))
      File "/usr/local/lib/python3.6/dist-packages/torch/nn/modules/module.py", line 722, in _call_impl
        result = self.forward(*input, **kwargs)
      File "/root/imprisonment_predict/model.py", line 63, in forward
        attention_mask=attention_mask)[0]
      File "/usr/local/lib/python3.6/dist-packages/torch/nn/modules/module.py", line 722, in _call_impl
        result = self.forward(*input, **kwargs)
      File "/usr/local/lib/python3.6/dist-packages/transformers/modeling_albert.py", line 682, in forward
        input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
      File "/usr/local/lib/python3.6/dist-packages/torch/nn/modules/module.py", line 722, in _call_impl
        result = self.forward(*input, **kwargs)
      File "/usr/local/lib/python3.6/dist-packages/transformers/modeling_albert.py", line 239, in forward
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    RuntimeError: The size of tensor a (1024) must match the size of tensor b (512) at non-singleton dimension 1
    
    ```

    - 试一下把输入的长度缩小

    - ```bash
      python preprocess.py --max-input-len 500
      python train.py --max-input-len 500
      ```

  - 果然是过长了

    - 变成别的错误了：

    - ```bash
      2020-11-18 16:29:15,882 Classify INFO: [2] Start training......
      Traceback (most recent call last):
        File "train.py", line 205, in <module>
          work(args)
        File "train.py", line 176, in work
          train(train_inputs, train_outputs, args, logger)
        File "train.py", line 77, in train
          loss = criterion(pred, label)
        File "/usr/local/lib/python3.6/dist-packages/torch/nn/modules/module.py", line 722, in _call_impl
          result = self.forward(*input, **kwargs)
        File "/usr/local/lib/python3.6/dist-packages/torch/nn/modules/loss.py", line 948, in forward
          ignore_index=self.ignore_index, reduction=self.reduction)
        File "/usr/local/lib/python3.6/dist-packages/torch/nn/functional.py", line 2422, in cross_entropy
          return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
        File "/usr/local/lib/python3.6/dist-packages/torch/nn/functional.py", line 2218, in nll_loss
          ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
      RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 'target' in call to _thnn_nll_loss_forward
      
      ```

      

    - 哦，原来是我大意了，pytorch 中计计算交叉熵损失函数时target的维度就是[batch_size]，这个pytorch不讲武德

  - epoch0测试的时候报错了

    - ```bash
      Traceback (most recent call last):
        File "train.py", line 205, in <module>
          work(args)
        File "train.py", line 176, in work
          train(train_inputs, train_outputs, args, logger)
        File "train.py", line 97, in train
          score = validate(model, device, args)
        File "train.py", line 138, in validate
          pred_prob = model(input_ids.to(device), segments_tensor.to(device), attention_mask.to(device))
        File "/usr/local/lib/python3.6/dist-packages/torch/nn/modules/module.py", line 722, in _call_impl
          result = self.forward(*input, **kwargs)
        File "/root/imprisonment_predict/model.py", line 61, in forward
          if (use_cls):
      RuntimeError: Boolean value of Tensor with more than one value is ambiguous
      ```

    - 原来是我的validate里面给model.forward传多了参数，之前改动forward的时候就应该警觉的

GPU加速训练

```
python train.py --batch_size 32
```



## test

需要根据官网给的公式来计算

TODO

最后要改下注释

TODO

## 疑惑

为啥albert的embedding只能接受长度小于512的input_ids ？

因为词向量的每个位置都编码了，大于512会出现2个词也有同一个编码(貌似是这样的)。