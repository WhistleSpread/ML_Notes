#-*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"

def backward(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    # 调用前向传播的数据，计算输出y
    y = mnist_forward.forward(x, REGULARIZER)
    # 给轮数计数器赋初值，设定为不可训练
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, lables = tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    # 调用包含正则化的损失函数 loss
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 定义学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, 
        global_step, 
        mnist.train.num_exapmles / BATCH_SIZE, 
        LEARNING_RATE_DECAY, 
        staircase=True, 
    )

    # 定义训练过程 
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

    # 定义滑动平均
    ema = tf.trian.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    
    # 实例化saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 在with 结构中初始化所有变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 在反向传播的过程中加载ckpt的操作，如果ckpt存在，则将ckpt restore 到saver中，恢复当前会话
        # 实现了给所有的w和b赋网络存在的ckpt中的值，实现断点连续，有了断点连续，就可以恢复上次训练的结果了
        # 再次训练的时候，程序会找到断点，继续从断点处开始运行， 实现了全连接网络的设计
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)



        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x:xs, y_:ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot = True)
    backward(mnist)

if __name__ == '__main__':
    main()
