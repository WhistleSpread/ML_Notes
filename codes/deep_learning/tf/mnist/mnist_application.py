"""
# 实现输入手写数字图片，输出识别结果
# 如何对输入的真实图片输出预测结果? 如何制作数据集，实现特定的应用
# 给我一堆标注过了的图片，我可以自己制作成数据集，实现特定的应用
# 例如将癌症图片作为数据集一起喂给神经网络，训练模型;
# 如何对输入的真实图片，输出预测结果？

def application():
    testNum = input("input the number of test pictures")
    for i in range(testNum):
        testPic = raw_input("the path of test picture:")
        # 将得到的图片对象处理成我们需要的向量形式
        testPicArr = pre_pic(testPic)
        # 通过传入arr到模型中，得到预测值
        preValue = restore_model(testPicArr)
        print("The prediction number is:", preValue)
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward

def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x:testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1 




# 实现对输入图片的手写数字识别

def pre_pic(picName):
    img = Image.open(picName)
    # 这里的参数 Image.ANTIALIAS 表示用消除锯齿的方法来进行resize
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    # 为了符合模型对颜色的要求(只有白和黑)，将reIm用convert变成灰度图，并见这个图转换为矩阵对形式
    im_arr = np.array(reIm.convert('L'))
    # 模型要求的是黑底白字，我们输入的图片是白底黑字，所以要给输入图片反色，使用嵌套循环遍历每个像素点
    threshold = 50
    for i in range(28):
        for j in range(28):
            # 反色
            im_arr[i][j] = 255 - im_arr[i][j]

            # 给图片做二值化处理，让图片只有纯白色点和纯黑色点，这样可以过滤掉手写数字中的噪声，留下图片主要特征，
            # 小于阈值的点认为是纯黑色，大于阈值的点认为是纯白色，可以自己适当调节阈值，让图像尽量包含手写数字的完整信息
            # 也可以尝试用其他方法来过滤掉噪声
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    
    # 整理形状为1行784列
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    # 将0-255之间的RGB值转换成0-1之间的浮点数
    img_ready = np.multiply(nm.arr, 1.0/255.0)
    # 完成了图片的预处理操作，符合神经网络对图像的输入要求了
    return img_ready

def application():
    testNum = input("input the number of test pictures: ")
    for i in range(testNum):
        testPic = raw_input("the patch of test pictures:")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print("The prediction number is: ", preValue)

def main():
    application()

if __name__ == '__main__':
    main()



