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

    