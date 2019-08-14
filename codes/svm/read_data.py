import numpy as np
import struct
import matplotlib.pyplot as plt

def load_images(file_name):
    ##   0表示不使用缓冲，1表示在访问一个文件时进行缓冲。
    binfile = open(file_name, 'rb') 
    buffers = binfile.read()
    magic,num,rows,cols = struct.unpack_from('>IIII',buffers, 0)
    # 整个images数据大小为60000*28*28
    bits = num * rows * cols
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    # 转换为[60000,784]型数组
    images = np.reshape(images, [num, rows * cols])
    return images

def load_labels(file_name):
    binfile = open(file_name, 'rb')
    buffers = binfile.read()
    # 读取label文件前2个整形数字，label的长度为num
    magic,num = struct.unpack_from('>II', buffers, 0) 
    # 读取labels数据
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    binfile.close()
    labels = np.reshape(labels, [num])
    return labels


filename_train_images = './data/train-images-idx3-ubyte'
filename_train_labels = './data/train-labels-idx1-ubyte'
filename_test_images = './data/t10k-images-idx3-ubyte'
filename_test_labels = './data/t10k-labels-idx1-ubyte'

train_images=load_images(filename_train_images)
# print(train_images)
train_labels=load_labels(filename_train_labels)
test_images=load_images(filename_test_images)
test_labels=load_labels(filename_test_labels)

# fig=plt.figure(figsize=(8,8))
# fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
# for i in range(30):
#     images = np.reshape(train_images[i], [28,28])
#     ax=fig.add_subplot(6,5,i+1,xticks=[],yticks=[])
#     ax.imshow(images,cmap=plt.cm.binary,interpolation='nearest')
#     ax.text(0,7,str(train_labels[i]))
# plt.show()