#### 4 Error back-propagation

误差反向传播，我们会使用梯度下降的方法来训练神经网络，我们有可能使用批量梯度下降，在批量梯度下降或者是随机梯度下降，为了让我们的符号简单一点，我们在下面使用的是随机梯度下降；

对于一个训练样本$(x, y)$我们对其要使用随机梯度下降，我们需要计算损失函数关于参数$\bold W$的梯度$\nabla_{\bold W}Loss(NN(\bold x;\bold W), \bold y)$；这里$\bold W$表示所有的权重$\bold W^l, W_0^l$, 在所有的层$l = (1, …, L)$, 这样看起来非常恐怖，因为所有层中的参数都要求梯度，这个太可怕了；但是如果使用链式法则的话，这个实际上还是蛮简单的；

回忆一下，we are always computing the gradient of the loss function *with respect to the weights* for a particular value of (x, y). That tells us how much we want to change the weights, in order to reduce the loss incurred on this particular training example. 

首先，我们来看一下，损失函数是如何受到最后一层的权重$\bold W^L$的影响的；我们知道我们最后的输出是$\bold A^L$, 我们用$Loss(\bold A^L, \bold y)$表示神经网络的损失函数，最后，$A^L = f^L(Z^L)$, 并且$Z^L = {W^L}^TA^{L-1}$, 这样以来我们就可以使用链式法则:
$$
\frac {\partial Loss}{\partial \bold W^L} = \frac {\partial Loss}{\partial \bold A^L} \cdot \frac {\partial \bold A^L}{\partial \bold Z^L}\cdot\frac {\partial \bold Z^L}{\partial \bold {W^L}}
$$

> 在这里使用链式法则的时候，你可能会对最后一项感到奇怪，事实上$\frac {\partial \bold Z^L}{\partial \bold {W^L}} = \frac {\partial \bold Z^L}{\partial \bold {W^L}^T} = \bold A^{L-1}$ 

我们来检验一下这里的矩阵的维度是不是匹配的，这个损失函数的求导，维数必须要是匹配的，对于每一层都要是匹配的，包括$l = L$:

$\frac {\partial Loss}{\partial \bold W^L}$ 是一个$m^l \times n^l$ 型的矩阵

$\bold A^{l-1}$是一个$m^l \times 1$ 型的矩阵

$(\frac {\partial loss} {\partial \bold Z^l })^T$ 是一个$1 \times n^l$的矩阵

> 这个地方你有没有想明白呢？可能这个地方想的好像不是很明白的样子！！！

所以，要找到损失函数的梯度，相对于网络中其他层的权重，我们所要做的就是要找到$(\frac {\partial loss} {\partial \bold Z^l })^T$ ;如果我们重复的使用链式法则的话，我们就可以得到损失函数对第一层的pre-activation 的求导；
$$
\frac {\partial loss}{\partial\bold Z^1 } = \frac {\partial loss }{\partial \bold A^L} \cdot \frac {\partial \bold A^L}{\partial \bold Z^{L}} \cdot \frac {\partial \bold Z^L}{\partial \bold A^{L-1}} \cdot \frac {\partial \bold A^{L-1}}{\partial \bold Z^{L-1}} ...... \frac {\partial \bold A^2}{\partial \bold Z^2} \cdot \frac {\partial \bold Z^2}{\partial \bold A^1} \cdot \frac {\partial \bold A^1}{\partial \bold Z^1} \cdot 
$$
这个偏导数的形式不那么正式，下面要给你看一下一般形式的计算结构。实际上，to get the dimensions to all work out， 我们必须反着写，我们先对这些量理解多理解一下；

- 







#### 5 Training

接下来就是使用随机梯度下降来训练feed-forward神经网络；我们用高斯分布来初始化参数，实际的梯度的计算并没有在我们的代码中直接的定义，因为我们想让我们的伪代码的结构显得简单一些；



对于参数$\bold W$的初始化非常的重要，如果初始化不好的话，那么神经网络的训练会非常的糟糕；将权重初始化为随机值是非常重要的，我们想要神经网络的不同部分来处理问题的不同方面；如果神经网络的所有权重都是从一个相同的数开始的话，the symmetry will often keep the values from moving in useful directions. 而且，当pre-activatioin 非常大的时候，在激活函数的梯度都趋近于0， so we generally want to keep the initial weights small so we will be in a situation where the gradients are non-zero, so that gradient descent will have some useful signal about which way to go.

One good general-purpose strategy is to choose each weight at random from a Gaussian (normal) distribution with mean 0 and standard deviation (1/m) where m is the number of inputs to the unit.

如果输入的$x$是一个为全1的向量，那么这个pre-activation 的值是啥？所有权重的和，期望值为0；





#### 6 Loss functions and activation functions

关于损失函数与最后一层的激活函数的选取，如果最后一层的激活函数是线性的，那么我们的损失函数就是二次损失函数，或者是hinge loss,如果最后一层的激活函数是sigmoid或者是softmax, 那么损失函数就是NLL,或者是NLLM；

##### 6.1 **Two-class classification and log likelihood**

对于二分类问题而言，hinge loss 给了我们一种方式来造一个更加平滑的目标函数，一个被广泛使用，并且具有良好的概率解释的损失函数并且这个损失函数可以很好的扩展到多分类问题当中，这个损失函数就是所谓的负对数似然函数，我们先会在二分类问题中讨论这个损失函数，然后再将其推广到多分类的问题当中；





















































