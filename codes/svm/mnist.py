import numpy as np
import svm
import kernal
import read_data as data

class MNIST_SVM:
    def __init__(self,data=[],label=[],C=0,toler=0,maxIter=0,**kernelargs):
        self.classlabel = np.unique(label)
        self.classNum = len(self.classlabel)
        self.classfyNum = (self.classNum * (self.classNum-1))/2
        self.classfy = []
        self.dataSet={}
        self.kernelargs = kernelargs
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        m = np.shape(data)[0]
        
        for i in range(m):
            if label[i] not in self.dataSet.keys():
                self.dataSet[label[i]] = []
                self.dataSet[label[i]].append(data[i][:])
            else:
                self.dataSet[label[i]].append(data[i][:])

    def train(self):
        num = self.classNum
        for i in range(num):
            for j in range(i+1,num):
                data = []
                label = [1.0]*np.shape(self.dataSet[self.classlabel[i]])[0]
                label.extend([-1.0]*np.shape(self.dataSet[self.classlabel[j]])[0])
                data.extend(self.dataSet[self.classlabel[i]])
                data.extend(self.dataSet[self.classlabel[j]])
                mnist_svm = svm.smoP(np.array(data),np.array(label),self.C,self.toler,self.maxIter,**self.kernelargs)
                mnit_svm.smoP()
                self.classfy.append(mnist_svm)
        self.dataSet = None

    def predict(self,data,label):
        m = np.shape(data)[0]
        num = self.classNum
        classlabel = []
        count = 0.0
        for n in range(m):
            result = [0] * num
            index = -1
            for i in range(num):
                for j in range(i + 1, num):
                    index += 1
                    s = self.classfy[index]
                    t = s.predict([data[n]])[0]
                    if t > 0.0:
                        result[i] +=1
                    else:
                        result[j] +=1

            classlabel.append(result.index(max(result)))
            if classlabel[-1] != label[n]:
                count +=1
                print(label[n],classlabel[n])
        #print classlabel
        print("error rate:",count / m)
        return classlabel
      
    def save(self,filename):
        fw = open(filename,'wb')
        pickle.dump(self,fw,2)
        fw.close()

    @staticmethod
    def load(filename):
        fr = open(filename,'rb')
        mnist_svm = pickle.load(fr)
        fr.close()
        return mnist_svm


def main():
    
    train_data,train_label = data.train_images, data.train_labels
    mnist_svm = MNIST_SVM(train_data, train_label, 200, 0.0001, 10000)
    mnist_svm.train()
    mnist_svm.save("mnist_svm.model")
    
    mnist_svm = LibSVM.load("mnist_svm.model")
    test,testlabel = data.test_images, data.test_labels
    mnist_svm.predict(test,testlabel)


main()