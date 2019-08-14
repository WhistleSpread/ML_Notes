# 一个简化版本的的SMO
"""
A simplified version of SMO
create an alphas vector filled with 0s
While the number of iterations is less than MaxIterations
    for Every data vector in the dataset:
        if the data vector can be optimized(这里是什么意思？什么叫可以被优化？):
            Select another data vector at random
            Optimize the two vectors together
            if the vectors can not be optimized -- break
    if no vectors were optimized --> increment the iteration count
"""
import random
import numpy as np
import copy


# helper functions for the SMO algorithm
def loadDataSet(fileName):
    # opens up the file and parses each line into class labels, and our data matrix.
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    # randomly selects one integer from a range
    """
    takes two values. 
    The first one, i, is the index of our first alpha, 
    and m is the total number of alphas. 
    A value is randomly chosen and returned as long as it’s not equal to the input i.
    """
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    # clip values if they get too big
    """
    clips alpha values that are greater than H or less than L. 
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

dataArr,labelArr = loadDataSet('testSet.txt')
# print('dataArr = ', dataArr)
# print('labelArr = ', labelArr)

"""
一个简单版本的SMO算法
Create an alphas vector filled with 0s
While the number of iterations is less than MaxIterations:
    For every data vector in the dataset:
        If the data vector can be optimized(什么叫做能够被优化啊？这个地方看不太明白):
            Select another data vector at random Optimize the two vectors together
            If the vectors can’t be optimized ➞ break        
    If no vectors were optimized ➞ increment the iteration count
"""

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    函数输入有5个参数:
    dataMatIn:输入数据集
    classLabels:类别标签
    C: 常数C
    toler: 容错率
    maxIter:退出前最大的循环次数
    """
    # 上述函数将多个列表和输入 参数转换成NumPy矩阵,为了后面简化数学操作，比如转置等
    dataMatrix = np.mat(dataMatIn)
    # 转置类别标签，得到的就是一个列向量而不是列表，类别标签向量的每行元素都和数据矩阵中的行一一对应
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    print('m, n=', m, n)
    # Create an alphas vector filled with 0s， 构建一个alpha列矩阵，矩阵中元素都初始化为0
    # print(np.mat(np.zeros((m, 1))))
    alphas = np.mat(np.zeros((m, 1)))
    # 该变量存储的是在没有任何alpha改变 的情况下遍历数据集的次数， 当该变量达到输入值maxIter时，函数结束运行并退出
    iter = 0

    while(iter < maxIter):
        # 每次循环当中，将alphaPairsChanged先设为0，然后再对整个集合顺序遍历， alphaPairsChanged用于记录alpha是否已经进行优化
        alphaPairsChanged = 0
        for i in range(m):
            # 实现:f(x) = labels*w^Tx +b， fXi能够计算出来，这就是我们预测的类别
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b
            # 基于这个实例的预测结果和真实结果labelMat[i]的比对，就可以计算误差Ei
            Ei = fXi - float(labelMat[i])

            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 如果误差很大，那么可以对该数据实例所对应的alpha值进行优化
                # 不管是正间隔还是负间隔都会被测试，所以这里有个 or, 
                # 也要同时检查alpha值，以保证其不能等于0或C
                # 由于后面alpha小于0或大于 C时将被调整为0或C，所以一旦在该if语句中它们等于这两个值的话，那么它们就已经在“边界” 上了，
                # 因而不再能够减小或增大，因此也就不值得再对它们进行优化了。
                # 这个if条件是不是就是表示一个数据向量能够被优化？ Enter optimization B if alphas can be changed

                # 如果能够被优化，那么随机选取一个在0～m中不等于i的随机数 Randomly select second alpha
                j = selectJrand(i,m)
                # 来计算这个alpha_j值的误差
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])

                # 稍后可以将新的alpha值与老的alpha值进行比较, 明确地告知Python要为alphaIold和alphaJold 分配新的内存
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # Guarantee alphas stay between 0 and C
                # 计算L和H,它们用于将alpha[j]调整到0到C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:
                    # 如果L和H相等，就不做任何改变，直 接执行continue语句
                    print("L==H") 
                    continue

                # Eta是alpha[j]的最优修改量, 如果eta为0，那就是说需要退出for循环的当前迭代过程
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                
                # 可以计算出一个新的alpha[j]
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                # 利用辅助函数clipAlpha以及L与H值对其进行调整。
                alphas[j] = clipAlpha(alphas[j],H,L)


                # 检查alpha[j]是否有轻微改变，如果是的话，就退出for循环，然后， alpha[i]和alpha[j]同样进行改变，
                # 虽然改变的大小一样，但是改变的方向正好相反(即如果一个增加，那么另外一个减少)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print ("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i]*(alphaJold - alphas[j])

                # 在对alpha[i]和alpha[j]进行优化之后，给这两个alpha值设置一个常数项b
                b1 = b - Ei- labelMat[i] * (alphas[i]-alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMat[j] * (alphas[j]-alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i] * (alphas[i]-alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMat[j] * (alphas[j]-alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T

                # 必须确保在合适的时机结束循环。如果程序执行到for循环的最后一行都不执行continue语句，那么就已经成功地改变了一对alpha，同时可以增加 alphaPairsChanged的值
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))

        if (alphaPairsChanged == 0): 
            iter += 1
        else: 
            iter = 0
        print("iteration number: %d" % iter)

    return b,alphas

b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
print("b = ", b)
