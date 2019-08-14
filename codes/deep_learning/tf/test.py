# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

def weight_variable(shape):
    # 生成一个截断正态分布
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 生成一个截断正态分布
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积层
def conv2d(x, W):
    """
    这里的2d表示它是一个二维的卷积操作
    x input a tensor of shape '[batch, in_height, in_width, in_channels]', 批次，图片的长和宽，通道数(黑白为1，彩色为3)
    W Filter / kernel tensor of shape '[filter_height, filter_width, in_channels, out_channels]'
    W 就是一个滤波器 / 卷积核， 滤波器的长和宽，输入的通道数和输出的通道数
    strides[0] = strides[3] = 1 必须都是1， strides[1] 代表x方向的步长，strides[2] 代表y方向的步长
    padding: a 'string' from: "VALID","SAME" 
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化层
def max_pool_2x2(x):
    """ 
    输入的x和上面的conv2d是一样的，
    ksize [1, x, y, 1] 表示窗口的大小，第0个和第3个位置都要设置为1，中间两个值代表窗口的大小
    strides 也是这样,第0个和第3个值都要是1，这里设置步长均为2；
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

lr = tf.Variable(0.0001, dtype=tf.float32)

# 在进行池化操作的时候，我们需要的是一个二维的数据，所以需要改变x的格式，转化为4D的向量[batch, in_height, in_width, in_channels]
# 也就是说，在这里就是把784复原为28*28
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一个卷积层的权值和偏置值
""" shape = [5*5], 输入是1个通道，输出是32个通道 32个卷积核从1个平面抽取特征，使用32个卷积核进行采样，得到32个特征平面"""
W_conv1 = weight_variable([5, 5, 1, 32])
""" 每一个卷积核设置一个偏置值 """
b_conv1 = bias_variable([32])

# 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数,得到第一个卷积层计算的结果h_conv1
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 初始化第二个卷积层的权值和偏置值
""" 注意这里传入的是32， 因为在进过第一个卷积层之后，输出的是32个特征图，所以在这里的输入就变成32个了 """
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# 把h_pool1权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

"""
28*28 的图片第一次卷积后还是28*28, 因为我们使用的padding的方式是same padding, 所以它的大小不会改变
第一次池化后变成了14*14
第二次卷积后为14*14，
第二次池化后变成了7*7
经过上面的操作，得到了64张7*7的平面
"""

# 初始化一个全连接层的权值
""" 上一层输出的有7*7*24，设置全连接层的神经元个数为1024个"""
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# 将池化层2的输出7*7*64扁平化为一维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# 求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# drop_out 操作
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 初始化第二个全连接层
# 因为前面的全连接层的输出是1024， 所以这里输入是1024， 输出是10,表示10个分类
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 计算输出：经过了drop_out 的全连接层的输出；
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交叉熵代价函数

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))                     
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
init = tf.global_variables_initializer()

# 将结果存放在一个bool类型的列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        # 在每个周期都对学习率进行赋初值,注意，在tensorflow中，赋初值都要用assign
        sess.run(tf.assign(lr, 0.001*(0.95**epoch)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})
            
        learning_rate = sess.run(lr)
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        print("Iter", str(epoch), "Testing Accuracy = ", str(acc), "Learning Rate = ", str(learning_rate))                      
