import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN


def img2vector(filename):
    """
    将32x32的二进制图像转换为1x1024向量
    因为每一个txt文件就是32x32的图片
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            # 因为returnVect 只有1行，所有每次行数都是0
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    #返回trainingDigits目录下的文件名
    trainingFileList = listdir('./data/digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)

        #将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = img2vector('./data/digits/trainingDigits/%s' % (fileNameStr))
    
    # 到此就得到了trainingMat
    neigh =KNN(n_neighbors = 3, algorithm = 'auto')
    neigh.fit(trainingMat, hwLabels)

    testFileList = listdir('./data/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)


    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])

        vectorUnderTest = img2vector('./data/digits/testDigits/%s' % (fileNameStr))
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))


if __name__=='__main__':
    handwritingClassTest()