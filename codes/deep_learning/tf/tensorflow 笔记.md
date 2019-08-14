tensor flow 教程

使用tensorflow之前一定要激活tensorflow环境

```
source activate tensorflow
```

最后退出环境也要

```
source deactivate
```



### 文件操作

```python
game_data = {
  "position":"N2 E3",
  "pocket":["keys","knife"],
  "money":160
}

import pickle

// 写入文件
save_file = open("save.dat", "wb")		// 文件变量 = open("文件路径名称", "wb")
pickle.dump(game_data, save_file)			// pickle.dump(待写入的变量， 文件变量)
save_file.close()											// 文件变量.close()

// 读取文件
load_file = open("save.dat", "rb")
load_game_data = pickle.load(load_file)
load_file.close()
```



```vim 不保存退出: esc:q!```



阶：张量的维数

一阶张量就是标量，二阶张量是向量，三阶张量是矩阵, 

| 维数     | 阶 | 名字    |例子|
| ---------------- | ------------ | ---------|------- |
| 0-D          | 0     | 标量 scalar | s = 123 |
| 1-D          | 1     | 向量 vector | v = [1, 2, 3]|
| 2-D          | 2     | 矩阵 matrix | m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]|
| n-D          | n     | 张量 tensor | s = [[[...]]]|

张量可以表示0阶到n阶数组(列表)

```
修改vim的配置 
vim ~/.vimrc 写入：
set ts = 4		// 表示一个tab键是4个空格
set nu				// 表示显示行号

:wq 表示保存退出
```



 

常见的激活函数有如下几个：

![avatar](/Users/gongmike/Desktop/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/pics/activation-function.png)

一个使用正则化的例子：

第三课 Tensorflow线性回归以及分类的简单使用，softmax介绍

第四课 交叉熵(cross-entropy)，过拟合，dropout以及Tensorflow中各种优化器的介绍。

第五课 使用Tensorboard进行结构可视化，以及网络运算过程可视化。
第六课 卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题。
第七课 递归神经网络LSTM的讲解，以及LSTM网络的使用。

第八课 保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别。
第九课 Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别。
第十课 多任务学习以及验证码识别。
第十一课 word2vec讲解和使用，cnn解决文本分类问题。
第十二课 语音处理以及使用LSTM构建语音分类模型。







1. 统计学习方法, 李航
2. 机器学习, 周志华
3. 深度学习
4. cs231n
5. 吴恩达机器学习
6. cs229
7. cs224n
8. 一学期的优达学城数据分析纳米学位
9. 林轩田机器学习基石+技法
10. 深度学习花书前两部分，毕设做的RNN和LSTM分类



论文

1. 读过rcnn，fast rcnn，faster rcnn，mask rcnn等论文

2. 读图像分类从AlexNet到DenseNet的论文，并实现了一部分，正在学习目标检测相关知识
3. 已读关于谱聚类论文,CNN,CNN for sentence,GAN,DCGAN的文章
4. 论文阅读：Efficient Backprop，提出Resnet和resnetV2的两篇论文，The power of depth for feedforward neural networks(为resnet提供了声援），Fixup Initialization：Residual learning without normalization



Kaggle

1. kaggle泰坦尼克号学做教程83%，自己的mlp实现63%(特征工程做的不好)



