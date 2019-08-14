#### 7 优化神经网络参数

神经网络都是一些参数函数，我们可以通过使用标准的梯度下降软件来通过对参数的优化来优化损失函数；但是，我们可以考虑利用损失函数的结构以及hypothesis class 来提升我们的优化性能；正如我们所看到的；the modular function-composition structure of a neural network hypothesis makes it easy to organize the computation of the gradient. 正如我们之前多看到的，损失函数的结构其实是一些和式，对于训练数据的每一个点都允许我们考虑随机梯度下降的方法，在这个章节，我们将会考虑一些其他的策略来组织我们的训练过程，使得处理step-size parameter 更加的容易；



##### 7.1 Batches

假设我们的目标函数是下面这种形式:
$$
J(W) = \sum_{i=1}^nL(h(x^{(i)}; W), y^{(i)})
$$
这里，$h$是神经网络所计算出来的函数，$\bold W$表示所有的权重矩阵，以及网络中的向量；

当我们使用batch gradient descent 的时候，我们使用下面的更新规则:
$$
\bold W := \bold W - \eta\nabla_{\bold W}J(\bold W)
$$
这个也等价于:
$$
\bold W := \bold W - \eta \sum_{i=1}^n\nabla_{\bold W}L((h(x^{(i)}; W), y^{(i)})
$$

因此，我们加总了在每个训练样本点损失的梯度(关于$\bold W$)的；然后朝着梯度的负方向走一步；

在随机梯度下降算法中，我们重复的选择一些点($x^{(i)}, y^{(i)}$), 这些点是我们从数据集中随机选择出来的一些点；我们对权重的更新只是基于一些单独的点：
$$
\bold W := \bold W - \eta \nabla_{\bold W}L((h(x^{(i)}; W), y^{(i)})
$$

只要我们在数据集中选出的点是uniformly at random， 学习率$\eta$取值合适的话，我们就可以保证有很大的可能至少会收敛到局部最优值；

这两个方法各自有各自的优缺点，The batch method takes steps in the exact gradient direction but requires a lot of computation before even a single step can be taken,especially if the data set is large. The stochastic method begins moving right away, and can sometimes make very good progress before looking at even a substantial fraction of the whole data set, but if there is a lot of variability in the data, it might require a very small η to effectively average over the individual steps moving in “competing” directions.

一种有效的策略就是综合一下batch 和 stochastic gradient descent， 通过使用一种叫做mini-batches的方法；我们选择mini-batches 的大小为k,我们选择k个distinct 数据点，这些点是uniformally at random 选出来的；我们对参数$\bold W$的更新是基于这些点对梯度的贡献来确定的:
$$
\bold W := \bold W - \eta \sum_{i=1}^k\nabla_{\bold W}L((h(x^{(i)}; W), y^{(i)})
$$
大多数的神经网络软件包都是使用的mini-batches的方法； 我们可以看到，如果说当k=1的时候，那么这样就相当于是随机梯度下降了，如果当k等于n的话，那么就是batch 梯度下降了；

要在一个非常的大数据集中挑选出k个unique 数据点，这个操作前在的计算复杂度是非常大的，一个可供选择的策略是：如果你有一种有效的方法来随机的shuffle你的数据集(或者随机shaffle这些数据集的索引)然后在一个循环里面进行操作，基本步骤如下：

Mini - batch -SGD (NN, data, k)

n = length(data)

while not done:

​		random-shuffle(Data)

​		for i=1 to n/k

​				batch - gradient - updata(NN, data[(i-1)k:ik])



##### 7.2 自适应的步长

选择参数$\eta$步长是很困难，并且很花费时间的，如果$\eta$选择的过小，那么收敛会比较慢，如果$\eta$选的过大，那么可能会由于震荡导致发散；这个问题在随机梯度下降或者是mini-batch 这些情况下显得尤为明显；因为我们知道我们需要减小步长，for the formal guarantees to hold;

在一个单一的神经网络中，这个也是正确的，我们可能非常想要不同的步长，因为，随着我们的网络变得越深，我们可以发现，在最后一层损失函数梯度的大小可能和第一层的损失函数的梯度的大小具有很大的差异，如果你看一下公式8.3,你就会发现， output gradient is multiplied by all the weight matrices of the network and is “fedback” through all the derivatives of all the activation functions. 这样一来就会导致梯度要么是爆炸式的增长，要么是梯度消失；在这种情况下，如果说我们的步长是固定不变的，那么反向传播的梯度就要么过大，要么过小；

因此，我们将会考虑对于每一个权重都有一个独立的步长，并且我们更新步长是基于a local view of how the gradient updates have been going.

> #### 因为前面的反向传播那个部分没有整明白，所以，这里也没有看的很清楚；





##### 7.2.1 滚动平均

我们首先看一下滚动平均的概念，滚动平均其实是对一系列的数据估计一个可能的平均权重值，我们假设我们的数据序列是$a_1, a_2, …$; 然后我们定义滚动平均值$A_0, A_1, A_2, …, $, 我们使用下面的这些等式:
$$
A_0 = 0 \\
A_t = \gamma_tA_{t-1} + (1 - \gamma_t)a_t
$$
在这里$\gamma_t \in (0, 1)$, 如果$\gamma_t$是一个常数，那么这个就是一个滑动平均值，在这个滑动平均值中:
$$
A_T = \gamma A_{T-1} + (1 - \gamma)a_T \\
= \gamma(A_{T-2} + (1 -\gamma)a_{T-1}) + (1 - \gamma)a_T \\
= \gamma A_{T-2} + \gamma(1  - \gamma)a_{T-1} + (1 - \gamma)a_T \\
 = \sum_{t = 0}^T \gamma^{T-t}(1-\gamma)a_t
$$
所以，你可以看到，输入值$a_t$,越接近于序列的尾部的值不在早前输入的值的影响要大，对于$A_t$, 如果我们将$\gamma_t = \frac {(t - 1)} {t}$ 那么我们最后得到的就是实际的平均值.

>第二个asertion 可以用归纳法来证明，这个并不难；



##### 7.2.2 Momentum

现在我们用一个叫做Momentum的方法，这种方法有点像滑动平均，因为它是使用滑动平均这种策略来计算$\eta$, 这种方法采取的策略是尽量“平均”最近的梯度更新，因此，如果它一直都是在某些方向上来回震荡的话，我们就取出运动的那个组成部分； so that if they have been bouncing back and forth in some direction, we take out that component of the motion.对于Momentum，我们有：
$$
\bold V_0 = 0 \\
\bold V_t = \gamma\bold V_{t-1} + \eta\nabla_{\bold W}J(\bold W_{t-1}) \\
\bold W_t = \bold W_{t-1} - \bold V_{t} \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ \  
$$
这个并不是很像一个自适应的步长，但是，我们可以看到，如果我们让$\eta = \eta'(1 - \gamma)$, 







##### 7.2.3 Adadelta



##### 7.2.4 Adam



#### 8 正则化

目前，我们只考虑了最优化损失函数，在训练数据上，作为我们的目标，对于神经网络，但是，正如我们之前讨论的那样，如果我们只这么做，有可能存在过拟合的现象，比较实际的做法是在当前的神经网络中(这个神经网络往往非常大，并且有许多数据要训练) 所以说过拟合并不是一个很大的问题，然而，这与我们目前的理论理解是背道而驰的，对这个问题的研究是一个热门的研究领域。换句话说，还是存在过拟合的情况的，然而，存在几种策略for regularizing a neural network, and they can sometimes be important.

##### 8.1 Methods related to ridge regression

有意思的是，有许多策略可以被证明是有相似的效果的：early stopping, weight decay, and adding noise to the training data.

提前停止这个策略被使用的非常的广泛，并且最容易实现；基本的思想是在你的训练集上进行训练，但是在每一个epoch(pass through the whole training set, or possibly more frequently)，但是在一个校验集上评价一下loss of the current $\bold W$, 





##### 8.2 Drop out





##### 8.3 Batch Normalization

















