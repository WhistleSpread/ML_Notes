import numpy as np
import math


def kernelTrans(X, A, kTup):
    m,n = np.shape(X)
    # 构建一个矩阵K,然后调用函数进行填充，如下一个是线性核，一个是径向基核
    # 这个矩阵为m*1的矩阵
    K = np.mat(np.zeros((m,1)))
    if kTup[0]=='lin':
        K = X * A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = math.exp(K /(-1*kTup[1]**2))
    else:
        raise NameError('We Have a Problem -- That Kernel is not recognized')
    return K