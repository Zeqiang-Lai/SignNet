# Log

2020-10-28(lzq)

- 新下了一个测试集[ASL Alphabet Test](https://www.kaggle.com/danrasband/asl-alphabet-test/home)
- 重新分了一个验证集出来8700张图
- 用现在代码训练10epoch左右即可在验证集上达到几乎100%的正确率。
- 但是测试机正确率很低：Test set: Average loss: 20.8044, Accuracy: 251/866 (29%)





## 遇到的问题

- 各个类别数据不平均
- 训练数据背景单一
- 有些很相近的手势很难识别正确
- 图片中手部大小对结果影响很大，a的手势近是a，远是s，O也是只有一定大小才能识别出来
- 手势旋转角度影响也很大，例如d的手势，转一下就变t
- C一定要露出三根手指
- 手势G放平就变成H了
- I太难识别了吧
- K有点难摆
- q和R基本识别不了，R会识别成U
- S的手势近看是E
- t老是识别成s
- v的手势近看是U
- W手势近看是B
- X手势和D容易混，很难识别
- Y要很远才能识别



## 可能有效的改进思路

- 数据增强：旋转（不行，有些手势旋转之后会变），仿射变换
- 对去背景后的图像再做一次去背景，可以用空间聚类
- 录视频保持基本满屏幕，然后数据增广的时候Padding再Resize改大小


## 注意事项

- 录制数据集的时候，注意保留原始彩色视频

## 训练指令

```shell
python train.py --train-set lzq_test_imgs --val-set lzq_test_imgs --val-interval 5 --epoch 30 --save-interval 5
python train.py --train-set custom_train_mask_img --val-set custom_test_mask_img --val-interval 2 --epoch 20 --save-interval 2
```

## 测试指令

```shell
python test.py --val-set custom_test_mask_img --resume_name Nov23_21-03-19_lzq-desktop
```

## TODO

报告：

- [ ] 3.SignNet：表格填好，网络结构图
- [ ] ASL Dataset和Test Dataset Preview 图
  - [ ] 录制一个测试集
- [ ] 重跑ASL实验，填好表格
- [ ] Custom Dataset
  - [x] 录制一个训练集
  - [x] 录制一个测试集
  - [ ] 写好剩余部分
  - [x] 制作一张示意图（和background subtraction用同一张就好), 
  - [ ] 29个符号的图
- [ ] Improvement的实验部分
  - [x] 准确率数值结果
  - [x] 混淆矩阵
  - [x] 帧率测试
- [ ] 摘要和conclusion



- [x] 训练数据集可能没有清理干净，导致S和E识别不好。

## 数据处理

```shell
python extract_img.py --input ../data/custom_train_mask --output ../data/custom_train_mask_img
python tmp_balance_data.py --input ../data/custom_train_mask_img --maxn 900

python extract_img.py --input ../data/custom_test_mask --output ../data/custom_test_mask_img
python tmp_balance_data.py --input ../data/custom_test_mask_img --maxn 300
```

## 命令

```shell
python live_demo.py --model experiments/custom/checkpoint_latest.pth --config experiments/custom/class_map.json
```

## 手语识别判定算法

- 连续一段时间都是一个字母，添加这个字母到缓冲区
- 之后如果还是这个字母不再添加
- 中间需要间隔一个Nothing
- 然后再连续一段都是一个字母，就再添加一个到缓冲区

