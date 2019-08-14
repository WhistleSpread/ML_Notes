感知机

首先，感知机这个名字非常的酷炫，感知机这个模型是基于McCulloch和Pitts以及Hebb在1943年构建出来的神经元模型。感知机模型由Rosenblatt于1962年开发。在当时，感知机只不过是作为一个算法出现，人们并没有试图将其用来优化任何特定的标准(particular criteria), 从感知机被提出来以后，大量的关于感知机收敛属性的分析以及其他方面的行为研究问世。

### 1. 算法

回忆一下，在上一讲中，我们训练的数据集$D_n$中的数据都是d维，$x \in \mathcal{R^d}$, 输出的$y \in \{-1, 1\}$, 感知机算法训练的是一个二分类分类器$h(x; \theta, \theta_0)$, 感知机算法通过$\tau$次迭代来找到$\theta$和$\theta_0$:

PERCEPTRON($\tau$, $D_n$)

$\theta = [0, 0, …, 0]^{T}$ <br>$\theta_0 = 0$<br>$for\ t=1 \ to \ \tau$<br>		$for\ i=1 \ to \ n$<br>				$if \ y^i(\theta^Tx^i + \theta_0) \le 0$<br>						$\theta = \theta + y^ix^i$<br>						$\theta_0 = \theta_0 + y^i$<br>return $\theta$, $\theta_0$



>可以检查一下这个算法中维数是否弄错，我们知道$\theta$是一个$d\times1$维的向量，$x^i$也是一个$d \times 1$维的列向量；$y^i$是一个标量；$\theta_0$是一个标量；稍微看一下就可以看到所有的都是匹配的。

从直觉上来讲，在每一步中，如果当前的假设$\theta$和$\theta_0$, 这个分类器将$x^i$分类正确了，那么就不会发生什么改变，但是如果$x^i$没有被分对，那么$\theta$以及$\theta_0$就会发生相应的变化；因此它会更加趋向于将$x^i， y^i$分对的模型；

要注意：如果这个算法遍历了数据集中所有的数据，假设模型$\theta$以及$\theta_0$都没有发生变化，那么之后它将不再会做任何更新，所以就应该在那一点终止；

> 可是在算法里面好像并没有体现这一点啊？并没有检查 i 是否到了 n ,如果 i 到了 n 的话就会跳出循环；
>
> 在不断的迭代过程中$\epsilon_n$也会不断的减小；

一个关于感知机很重要的事实就是：如果存在一个线性分类器能够做到完全分类的话，那么这个算法最终一定会找到这个分类器！我们之后会有详细的证明；



### 2. 偏移量

有时候，如果分类器是下面这种形式的话，更加容易去应用和分析
$$
h(x;\theta) =
\begin{cases}
+1 & if \ \theta^Tx > 0\\
-1 & otherwise
\end{cases}
$$


现在的情况是没有一个明确的偏移量$\theta_0$, 所以这个分类器必须穿过原点，所以使用这个分类器或多或少对我们的分类会产生限制；然而，我们可以将任何需要用到偏移量的线性分类器转换成无偏移量的分类器，我们所要做的就是提升到高维空间中来讨论这个问题。

考虑一个d维的线性分类器，这个线性分类器的垂直方向向量为$\theta = [\theta_1, \theta_2, …, \theta_d]$, 相应的偏移量为$\theta_0$; 对于每一个属于数据集的点$x \in \mathcal{D}$， 我们在这些点的向量后面加上一个+1,那么新的点在坐标就是:
$$
x_{new} = [x_1, x_2, ..., x_d, +1]^T
$$
然后我们定义
$$
\theta_{new} = [\theta_1, ..., \theta_d, \theta_0]
$$
那么就有：
$$
\theta_{new}^T \cdot x_{new} = \theta_1x_1 + \theta_2x_2 + ... + \theta_dx_d + \theta_0 \cdot 1 \\
= \theta^Tx + \theta_0 \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
$$
因此，这里的$\theta_{new}$就等价于一个$d+1$维的分类器(能够对我们原来的问题进行分类)，但是并没有偏移量，因为这里只是$\theta_{new}^T \cdot x_{new}$

现在考虑一下下面的数据集：
$$
X = [[1], [2], [3], [4]] \\
Y = [[+1], [+1], [-1], [-1]]
$$
这个数据集在一个一维空间中是线性可分的，我们只需要将参数$\theta = [-1]$ 并且$\theta_0 = 2.5$, 但是如果我们要求这个分类器必须穿过原点，那么我们就没办法找到一个这样穿过原点的分类器将他们分开；

> 因为如果要求分类器必须穿过原点，那么也就是说分类器$\theta^Tx + \theta_0$中的$\theta_0 = 0$, 我们必须找到一个$\theta^Tx$将这个数据集分类，可以看到，$X$全部为正数，$Y$有正有负，所以我们并没有办法找到这样一个$\theta$将其分分开；

那么接下来我们考虑将数据集中的数据都提高一个维度，在更加高维的空间中来讨论这个分类问题，我们构造一个新的数据集，这个新的数据集中数据长这个样子：
$$
X_{new} = 
\begin{bmatrix} 
\begin{bmatrix}
1 \\
1
\end{bmatrix}
\begin{bmatrix}
2 \\
1
\end{bmatrix}
\begin{bmatrix}
3 \\
1
\end{bmatrix}
\begin{bmatrix}
4 \\
1
\end{bmatrix}
\end{bmatrix}
$$
也就是说，我们将它的维度拉长了一维，拉长的那个维度我们都填充为1，那么这个新的数据集：
$$
X_{new} = 
\begin{bmatrix} 
\begin{bmatrix}
1 \\
1
\end{bmatrix}
\begin{bmatrix}
2 \\
1
\end{bmatrix}
\begin{bmatrix}
3 \\
1
\end{bmatrix}
\begin{bmatrix}
4 \\
1
\end{bmatrix}
\end{bmatrix}  \\
Y = [[+1], [+1], [-1], [-1]]
$$
我们就可以找到一个通过原点的超平面来将数据集分开；我们使用的$\theta_{new} = [-1, 2.5]^T$

> 其实感觉这个就是相当于是问题的转换，将原来的d维度的问题转换一下，转化为求解d+1维度的问题，只不过这个d+1维度的问题更加容易求解，并且得到的解刚好能够解决前面的$d$维的问题，在这里我们可以看到在d+1维的求的的通过原点的分类器的解刚好针对的就是d维不通过原点的问题的解

所以，如果我们要求我们的感知机算法必须严格经过原点的话，那么我们的感知机算法可以修改成如下的形式(之后我们会详细的研究这个算法)



PERCEPTRON-THROUGH-ORIGIN($\tau$, $D_n$)

$\theta = [0, 0, …, 0]^{T}$ <br>$for\ t=1 \ to \ \tau$<br>		$for\ i=1 \ to \ n$<br>				$if \ y^i(\theta^Tx^i + \theta_0) \le 0$<br>						$\theta = \theta + y^ix^i$<br>return $\theta$, $\theta_0$



### 3. 感知机理论

现在，我们会比较严肃的讨论一下感知机算法的分类效果到底如何。我们首先要说明一下：能够用感知机算法实现完美分类的问题都具有什么样的特点，然后证明为什么具有这样特点的问题能够用感知机做到完美的分类；除此之外，我们还会给出一个概念，这个概念是关于到底是什么使得感知机算法难以分类；并且将这个概念与算法迭代的次数联系起来；

#### 3.1 线性可分

一个数据集$\mathcal{D_n}$是线性可分的，当且仅当存在$\theta$,$\theta_0$, 使得对于所有的$i = 1, 2, …, n:$
$$
y^{(i)}(\theta^Tx^{(i)} + \theta_0) > 0
$$
换句话说，就是要所有在训练集上的预测都要能预测对,也就是说
$$
h(x^{(i)};\theta, \theta_0) = y^{(i)}
$$
再换句话说，训练误差要为0:
$$
\epsilon_n(h) = 0
$$

> 所谓的线性可分就是指，使用一个线性分类器，可以将训练集上的数据全部都分对；



#### 3.2 收敛理论

关于感知机得到的的基本结果就是：如果你的训练数据$\mathcal{D_n}$是线性可分的，那么感知机算法就一定能够保证找到一个线性分类器。

> 如果训练的数据不是线性可分的，那么这个算法将不能在有限的时间内告诉你说这个问题是线性不可分的，当然了，有一些其他的算法能够在有限的时间内(O($n^{d/2}$), O($d^{2n}$)或者是$O(n^{(d-1)} \log n)$)告诉你一个算法到底是不是线性可分的

如果我们要具体的度量一下一个数据集的线性可分性的话，我们可以用分类器的边界来进行度量；我们首先来定义一个点到一个超平面的margin：

首先回忆一下，一个点$x$到一个超平面$\theta$, $\theta_0$，的距离是:
$$
\frac {\theta^Tx + \theta_0} {||\theta||}
$$
那么我们就可以定义一个标号点$(x,y)$相对于超平面$\theta, \theta_0$的margin 就是：
$$
y \cdot \frac {\theta^Tx + \theta_0} {||\theta||}
$$
当且仅当这个点$x$被超平面超平面$\theta, \theta_0$分类器分类为y的时候这个量为正

> 也就是说，被分对了，这个margin 才是正的，否则(被分错了)这个margin 就是负的；

现在，相对于一个超平面分类器$\theta, \theta_0$分类器, 一个数据集$\mathcal{D_n}$的margin 就是所有点到这个超平面分类器的margin中的最小值；
$$
\min_\limits{i}(y^{(i)} \cdot \frac {\theta^Tx^{(i)} + \theta_0} {||\theta||})
$$
这个margin是正的，当且仅当在数据集中的所有的点都被正确的分类了，并且只有在这种情况下，这个margin 才反映了一个超平面到它最近的点的距离；



##### 定理 3.1(感知机收敛理论)

为了简化分析，我们首先考虑这个线性分类器必须通过原点的情况。考虑一个通过原点的分类器，如果有如下的条件成立：

a. 存在$\theta^*$, 使得$y^{(i)} \cdot \frac {\theta^Tx^{(i)} + \theta_0} {||\theta||} \ge \gamma$ 对于所有的$i = 1, …, n$并且对某些$\gamma > 0$

> 我不明白为什么这里是某些$\gamma > 0$?

b. 所有的测试样本的大小都是有界的，bounded magnitude: $||x^i \le R||$ 对于所有的$i = 1, …, n$；

那么感知机算法最多犯$(\frac R \gamma)^2$次错误，就能收敛到正确的分类器上。

证明：

我们首先初始化$\theta^{(0)} = 0$, 并且让$\theta^{(k)}$表示在感知机算法以及弄错了k次后的那个超平面(注意因为这里我们暂时只考虑通过原点的分类器，所有就没有偏移量$\theta_0$,我们现在要考虑一下在我们目标得到的这个超平面$\theta^{(k)}$与理想的目标分类器$\theta^*$之间的夹角；因为这两个分类器都是通过原点的，如果我们可以证明他们之间的夹角通过不断的迭代在不断的减小，那么最终我们就会收敛到我们所期望的那个目标分类器$\theta^*$;

因此，我们就来考虑一下这两个分类器之间的夹角，回忆一下关于点乘的定义
$$
cos(\theta^{(k)}, \theta^*) = \frac {\theta^{(k)} \cdot \theta^*} {{||\theta^*||}{||\theta^{(k)}||}}
$$
我们可以将等式右边的式子分成两个部分。
$$
cos(\theta^{(k)}, \theta^*) = (\frac {\theta^{(k)} \cdot \theta^*} {{||\theta^*||}}) \cdot (\frac 1 {||\theta^{(k)}||})
$$
我们首先将注意力放在第一个因子：

为了不使一般性，我们假设我们目前得到的第k个不能完全正确分类的分类器$\theta^k$将第$i$个样本$(x^{(i)}, y^{(i)})$给分错了. 因为是第$i$个点分类错误，根据感知机算法的操作：$\theta = \theta + y^ix^i$, 也就是说$\theta^{(k)} = \theta^{(k-1)} + y^ix^i $ 于是有：
$$
\frac {\theta^{(k)} \cdot \theta^*} {{||\theta^*||}} = \frac {(\theta^{(k-1)} + y^{(i)}x^{(i)}) \cdot \theta^*} {{||\theta^*||}} \\
= \frac {\theta^{(k-1)} \cdot \theta^*} {{||\theta^*||}} + \frac {y^{(i)}x^{(i)}) \cdot \theta^*} {{||\theta^*||}}
$$
根据我们前面的假设条件a:
$$
\frac {y^{(i)}x^{(i)}) \cdot \theta^*} {{||\theta^*||}} \ge \gamma
$$


所以对于公式16有
$$
\frac {\theta^{(k-1)} \cdot \theta^*} {{||\theta^*||}} + \frac {y^{(i)}x^{(i)}) \cdot \theta^*} {{||\theta^*||}} \ge \frac {\theta^{(k-1)} \cdot \theta^*} {{||\theta^*||}} + \gamma
$$
我们很容易发现公式18右边的$\frac {\theta^{(k-1)} \cdot \theta^*} {{||\theta^*||}}$与$\frac {\theta^{(k)} \cdot \theta^*} {{||\theta^*||}}$，的联系，于是我们使用数学归纳法就可以得到：
$$
\frac {\theta^{(k-1)} \cdot \theta^*} {{||\theta^*||}} + \frac {y^{(i)}x^{(i)}) \cdot \theta^*} {{||\theta^*||}} \ge \frac {\theta^{(k-1)} \cdot \theta^*} {{||\theta^*||}} + \gamma \ge \frac {\theta^{(k-2)} \cdot \theta^*} {{||\theta^*||}} + 2\gamma \ge ... \ge k\gamma
$$
现在我们来看看公式15右边的第二个因子，我们注意到，因为样本点$(x^{(i)}, y^{(i)})$是被错误分类的样本点，所以，也就是说$y^{(i)}({\theta^{(k-1)}}^Tx^{(i)}) \le 0$, 因此就有：
$$
{||\theta^{(k)}||}^2 = {||\theta^{(k-1)} + y^ix^i||}^2 \\
= {||\theta^{(k-1)}||}^2 + 2y^{(i)}({\theta^{(k-1)}}^Tx^{(i)}) + {||x^{(i)}||}^2  \\
\le {||\theta^{(k-1)}||}^2 + R^2 \le kR^2
$$
这个公式的推导也只是用了一下平方项的展开，运用了一下上面的条件b,有界，以及$y^{(i)}({\theta^{(k-1)}}^Tx^{(i)}) \le 0$,通过简单的递推关系我们就得到了最终结果；现在回到点乘的定义：
$$
cos(\theta^{(k)}, \theta^*) = (\frac {\theta^{(k)} \cdot \theta^*} {{||\theta^*||}}) \cdot (\frac 1 {||\theta^{(k)}||}) \ge k\gamma \cdot \frac 1 {\sqrt k R} = \sqrt k \cdot \frac \gamma R
$$
因为cos的值最多为1， 所以我们有：
$$
1 \ge \sqrt k \cdot \frac \gamma R \\
k \le {(\frac R \gamma)}^2
$$
这个结果赋予了数据集$\mathcal{D_n}$的margin$\gamma$以实际的可操作的意义，当我们在使用感知机算法来分类的时候，最多只需要试错${(\frac R \gamma)}^2$次，在这里，$R$ 表示 upper bound on the magnitude of the training vectors.

