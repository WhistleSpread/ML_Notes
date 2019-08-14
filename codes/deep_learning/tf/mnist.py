import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/', one_hot = True)

# 返回各个子集的样本数
print("train_data_size:", mnist.train.num_examples)
print("validation_data_size:", mnist.validation.num_examples)
print("test_data_size:", mnist.test.num_examples)

# 返回标签和数据
print(mnist.train.lables[0])
print(mnist.train.images[0])

# 取一小撮数据，喂入神经网络中
BATCH_SIZE = 200 # 定义一小撮是多少
xs, ys = mnist.train.next_batch(BATCH_SIZE)
print('xs_shape = ', xs.shape)
print('ys_shape = ', ys.shape)

# 几个常用的函数
# tf.get_collection(" ") # 从集合中取出全部变量，生成一个列表
# tf.add_n([]) # 列表内对于元素相加
# tf.cast(x, dtype) # 把x转换为dtype类型
# tf.argmax(x, axis) # 返回最大值所在索引号
# os.path.join("home", "name") # 返回home/name
# with tf.Graph().as_default() as g: # 其内定义的节点在计算图g中

# 保存模型
saver = tf.train.Saver()            # 实例化saver对象
# 每隔一定的轮数，将模型保存起来
with tf.Session() as sess:
    for i in range(STEPS):
        if i % 轮数 == 0：
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

# 加载模型
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state("modelpath")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)


# 实例化可还原滑动平均值的saver
ema = tf.train.ExponentialMovingAverage("base")
ema_restore = ema.variables_to_restore()
saver = tf.train.Saver(ema_restore)

# 准确率计算方法
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))