#-*- coding:utf-8 -*-
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight(shape, regularizer):
    # 随机生成参数w
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        # 将正则化项加入到总损失losses中
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros())
    return b

def forward(x, regularizer):
    # 搭建网络，描述从输入到输出到数据流
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    # y = tf.nn.relu(tf.matmul(y1, w2) + b2) 错了，这里的输出不过relu函数的
    y = tf.matmul(y1, w2) + b2
    # 这个结果是直接输出的，因为要对输出使用softmax函数，使它符合概率分布，所以输出y,不过relu函数
    return y