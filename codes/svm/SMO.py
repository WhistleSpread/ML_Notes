import numpy as np
import random
import math

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

# main components of optimization Structure 
class optStruct:
    """
    建立一个数据结构用来保存所有重要的值， 作为一个数据结构来使用对象
    将值传给函数时，我们可以通过将所有数据移到一个结构中来实现，这样就可以省掉手工输入的 麻烦了
    """
    def __init__(self, dataMatIn, classLabels, C, toler):
        # 构造函数：该方法可以实现其成员变量的填充
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        # 误差缓存，一个m×2的矩阵成员变量eCache
        # eCache的第一列给出的是eCache是否有效的标志位，而第二列给出的是实际的E值
        self.eCache = np.mat(np.zeros((self.m,2)))
    
    def calcEk(self, oS, k):
        # 辅助函数calcEk()能够计算E值并返回, 传入的是一个最优化的结构
        fXk = float(np.multiply(oS.alphas,oS.labelMat).T * (oS.X*oS.X[k,:].T)) + oS.b
        Ek = fXk - float(oS.labelMat[k])
        return Ek
    
    def selectJ(self, i, oS, Ei):
        # 内循环中的启发式算法
        # selectJ()用于选择第二个alpha或者说内循环的alpha值
        # 程序会在所有的值上进行循环并选择其中使得改变最大的那个值
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        oS.eCache[i] = [1,Ei]
        validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]

        if (len(validEcacheList)) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = oS.calcEk(oS, k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    # 选择具有最大步长的j
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = selectJrand(i, oS.m)
            Ej = oS.calcEk(oS, j)
        return j, Ej
    
    def updateEk(self, oS, k):
        # 计算误差值并存入缓存当中。在对alpha值进行优化之后会用到这个值
        Ek = oS.calcEk(oS, k)
        oS.eCache[k] = [1,Ek]


    # 用于寻找决策边界的优化代码，完整的SMO算法
    def innerL(self, i, oS):
        Ei = oS.calcEk(oS, i)

        if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
            j,Ej = oS.selectJ(i, oS, Ei)
            alphaIold = oS.alphas[i].copy()
            alphaJold = oS.alphas[j].copy()

            if (oS.labelMat[i] != oS.labelMat[j]):
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L==H: 
                # print("L==H") 
                return 0

            eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T

            if eta >= 0:
                # print("eta>=0")
                return 0
            oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
            oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
            oS.updateEk(oS, j)

            if (abs(oS.alphas[j] - alphaJold) < 0.00001):
                # print("j not moving enough")
                return 0

            oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
            oS.updateEk(oS, i)

            b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold) * oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j] * (oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
            b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold) * oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j] * (oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T

            if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
            elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
            else: oS.b = (b1 + b2)/2.0
            return 1
        else: return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
    iter = 0

    entireSet = True; 
    alphaPairsChanged = 0

    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += oS.innerL(i, oS)
            # print("fullSet, iter: %d i:%d, pairs changed %d" %(iter,i,alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += oS.innerL(i, oS)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        # print("iteration number: %d" % iter)

    return oS.b,oS.alphas


# dataArr,labelArr = loadDataSet('testSet.txt')
# print('dataArr = ', dataArr)
# print('dataArr.type = ', type(dataArr))
# b,alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
# print('b = ', b)
# print('alphas = ', alphas)