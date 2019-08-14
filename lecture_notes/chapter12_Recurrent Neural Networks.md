chapter12_Recurrent Neural Networks

在第8章中，我们学习了神经网络模型，我们知道了如何基于数据来训练一个网络的权重，这样的一个有权重的网络其实是一个函数，这个函数其实就是对训练集中数据对拟合；在第10章序列化模型中，我们学习了状态机模型，并且将循环神经网络定义为一个特定类型的状态机模型， 在这个状态机中，是用一个多维度的向量表示状态；在这一章节中，我们将会看到如何使用梯度下降的方法来训练一个循环神经网络的权重；so that it performs a *transduction* that matches as closely as possible a training set of input-output sequences.

### 1. RNN 模型

回忆一下，一个状态机的基本操作是：首先从某个状态$s_0$除法，然后不断的迭代的计算下面的式子:
$$
s_t = f(s_{t-1}, x_t) \\
y_t = g(s_t)
$$
正如下面的这张图所示(要记住这张图中在循环反馈回去的是需要有延迟的，否则就乱了套了)所以，如果非定了一系列的输入值$x_1, x_2, ...$ 那么这个状态机就会产生一系列的输出：
$$
y_1 = g(f(x_1,s_0)) \\
y_2 = g(f(x_2, f(x_1, s_0))) \\
......
$$

> 状态的转移是通过f函数的；

一个循环网络模型其实就是一个A *recurrent neural network* is a state machine with neural networks constituting functions  f and g 
$$
f(s,x) = f_1(W^{sx}x + W^{ss}s + W_0^{ss}) \\
g(s) = f_2(W^Os + W_0^O)
$$
在这个RNN模型当中，输入，输入，以及状态都是向量；在这里$x_t$是一个$\ell \times 1$的向量；$s_t$是一个$m \times 1$的向量；$y_t$是一个$v \times 1$的向量；

在网络中的权重值如下：
$$
W^{sx}:m \times \ell \\
W^{ss}: m \times m \\
W_0^{ss}: m \times 1 \\
W^O : v \times m \\
W_0^O : v \times 1
$$
在这里，我们的激活函数为$f_1$和$f_2$, 最后，RNN的操作被描述为如下方程：
$$
s_t = f_1(W^{sx}x_t + W^{ss}s_{t-1} + W_0) \\
y_t = f_2(W^Os_t + W_0^O)
$$

>在这里，其实我们可以检查一下矩阵的维数：
>
>要注意的是，在这里，我们对每一个输入，状态，都使用了函数；



### 2 Sequence-to-sequence RNN

现在我们可以来训练我们的RNN模型，这个RNN模型可以对一个序列的transduction进行建模；这个问题，有的时候被称为是序列到序列的映射。你可以把这个问题想成是一种回归问题模型：考虑一个输入的序列；学习产生出相应的输出序列；

一个训练集的形式如下:$[(x^{(1)}, y^{(1)}), ..., (x^{(q)}, y^{(q)})]$ 在这个训练集中：

- $x^{(1)}$和$y^{(1)}$都是长度为$n^{(q)}$的序列；
- 在一个pair里的两个序列的长度是相同的；而不同的pair里面的序列的长度则可能不同；

接下来，我们需要一个损失函数；我们首先定义一个在序列上的损失函数；要在这个序列上定义一个损失函数，其实存在许多可能的选择，一个通常的做法是将所有的输出值的每元素损失函数都加起来；
$$
Loss_{seq}(p^{(i)}, y^{(i)}) = \sum_{j=1}^{n(q)} Loss_{elt}(p_j^{(i)}, y_j^{(i)})
$$
对于每一个元素的损失函数$Loss_{elt}(p_j^{(i)}, y_j^{(i)})$，将取决于$y_t$的类型，以及它的编码的信息；就如我们的监督学习的网络中一样，然后我们让我们网络的中的参数为$(W^{sx}, W^{ss}, W_0^{ss}, W^O , W_0^O)$ 而我们的目标函数就是要最小化:
$$
J(\theta) = \sum_{i=1}^qLoss_{seq}(RNN(x^{(i)};\theta), y^{(i)})
$$
在这里，$RNN(x;\theta)$就是在给定输入序列为x的情况下循环神经网络所产生的序列；一般情况下，我们选择$f_1$的函数为$tanh$函数，对于$f_2$的函数，则是根据情况来定；

### 3 随着时间的反向传播

接下来就是有意思的事情了，我们可以使用梯度下降的方法通过寻找参数$\theta$来最小化目标函数$J$, 我们会使用最简单的方法:BPTT，随着时间的反向传播，一般来讲，这种方法并不是最好的方法，但是这种方法理解起来相对简单；在下面的第5部分中，我们会看一下，我们一般经常使用的优化方法。

BPTT的过程如下：

1. 对训练序列$(\bold x, \bold y)$进行采样，让这个序列的长度为$n$; 

2. "Unroll"这个RNN模型成为一个长度为n的序列；并且初始化状态为$s_0$: 

   > 现在，我们可以将我们的问题看成是在前馈神经网络中执行几乎普通的反向传播训练过程之一，但是和前馈神经网络不同的是，权重矩阵在各个层之间是共享的；在许多方面，这个类似于卷积神经网络中的操作；只不过在卷积网络中，这个权重是在空间上被重复使用，而在循环神经网络中，权重矩阵是在时间上被重复使用；

3. 通过使用前向传播的过程，来计算出预测输出的序列$p$:
   $$
   z_t^1 = W^{sx}x_t + W^{ss}s_{t-1} + W_0 \\
   s_t = f_1(z_t^1) \\
   z_t^2 = W^Os_t + W_0^O \\
   p^t = f_2(z_t^2)
   $$

4. 通过反向传播来计算梯度；在求权重矩阵的梯度的时候，$W^{ss}$和$W^{sx}$我们都要考虑；我们需要去找到:
   $$
   \frac {dL{seq}} {dW} = \sum_{u=1}^n \frac {dL_u} {dW}
   $$
   在这里，我们让$L_u = L_{elt}(p_u, y_u)$, 并且在这里使用全微分；and using the *total derivative*, which is a sum over all the ways in which W affects Lu，然后我们就有：
   $$
   \frac {dL{seq}} {dW} = \sum_{u=1}^n \frac {dL_u} {dW} = \sum_{u=1}^n\sum_{t=1}^n\frac {\partial L_u} {\partial s_t}\cdot \frac {\partial s_t}{\partial W}
   $$
   

   重新整理一下，我们就可以得到：
   $$
   = \sum_{t=1}^n \frac {\partial s_t}{\partial W} \cdot \sum_{u=1}^n\frac {\partial L_u} {\partial s_t}
   $$
   又因为$s_t$只会影响$L_t, L_{t+1}, ..., L_n$, 
   $$
    = \sum_{u=t}^n\frac {\partial L_u} {\partial s_t}\cdot \sum_{t=1}^n \frac {\partial s_t}{\partial W}
   $$
   ?????



### 4 训练一个语言模型

一个语言模型是在一组输入序列($c_1^{(i)}, c_2^{(i)}, ..., c_{n^i}^{(i)}$)上得到训练的;这个得到的语言模型被用来预测下一个character;当然是在给定了一组之前的tokens的前提下；

> 这里所说的token是指的一个character,或则一个word

$$
c_t = RNN(c_1, c_2, ..., c_{t-1})
$$

我们可以将这个问题转换成为一个序列到序列的训练模型；我们可以构建数据集$(x, y)$序列对；在这个序列对中，我们可以构建新的特殊的tokens；start and end, 用来标注一个序列的开头和结束的位置；



### 5 **Vanishing gradients and gating mechanisms**







一个能够使得RNN在长序列中表现良好的非常重要的insight，就是关于gating的概念；













































