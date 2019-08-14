from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1, 0.9], [0, 0], [0.1, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables


def classify0(inX, dataSet, labels, k):
    
    dataSetSize = dataSet.shape[0]
    # print("dataSetSize = ", dataSetSize)


    # tile 是numpy中的一个函数，tile（）函数内括号中的参数代表扩展后的维度，
    # 而扩展是通过复制A来运作的，最终得到一个与括号内的参数（reps）
    # 维度一致的数组(矩阵）tile(A, reps)

    DiffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # print("tile = ", tile(inX, (dataSetSize, 1)))
    # print("dataset = ", dataSet)
    # print("DiffMat = ", DiffMat)

    sqDiffMat = DiffMat**2
    # print("sqDiffMat = ", sqDiffMat)

    sqDistances = sqDiffMat.sum(axis=1)
    # 加入axis=1以后就是将一个矩阵的每一行向量相加
    # print("sqDistances = ", sqDistances)

    distances = sqDistances**0.5  
    # 到这里求解了欧式距离(并构成了一个ndarray)
    # print("distances = ", distances)

    sortedDistances = distances.argsort() # 根据排名作为索引 Index
    # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引号) 
    # print("sortedDistances = ", sortedDistances)

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        print("voteIlabel = ", voteIlabel)
        print("classCount.get(voteIlabel, 0) = ", classCount.get(voteIlabel, 0))
        #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    
    print("classCount =", classCount)
    
    # 选出了距离最小的k个点，并且做了一个简单的统计

    #python3中用items()替换python2中的iteritems()
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) 
    print("sortedClassCount = ", sortedClassCount)
    #按照第一个(从0开始数)进行排序
    return sortedClassCount[0][0] 
    # 返回的出现次数最多的那个标签


# 函数main:
group, labels = createDataSet()
x = [1, 1]
print(classify0(x, group, labels, 2))