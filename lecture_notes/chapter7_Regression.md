之前讲了机器学习中的二分类问题，现在转入到机器学习中的另外一个主题：回归；回归依然是监督学习，所以我们的数据的形式仍然是像下面这样：
$$
S_n = \{
(x^{(1)}, y^{(1)}),..., (x^{(n)}, y^{(n)}) 
\}
$$
但是现在这个y值它不再是一个离散的值了，$y$值都是实数值，因此我们的hypothesis的形式如下:
$$
h: \mathcal{R^d} \rightarrow \mathcal{R}
$$
当我们想要预测一个连续的数值量的时候，使用回归模型是一个不错个选择，比如说我们要回归一下身高，股票价格等等；在这里回归要区别于之前我们做的分类的问题；

做回归模型，第一步就是要选出一个损失函数，这个损失函数用来描述我们的hypothesis作出预测的效果；我们将我们的预测值和数据集中的实际值y做一个比较，对于损失函数的选择是你建模主要问题的一部分；在关于回归问题没有额外信息的情况下，我们一般都是使用的平方误差 squared error SE
$$
Loss(guess, actual) = (guess - actual)^2
$$
It penalizes guesses that are too high the same amount as it penalizes guesses that are too low, and has a good mathematical justification in the case that your data are generated from an underlying linear hypothesis, but with Gaussian-distributed noise added to the y values.

我们将会考虑下面这样一个linear hypothesis class：
$$
h(x;\theta, \theta_0) = \theta^Tx + \theta_0
$$
要记住，我们是实际上可以在进行线性回归之前先进行非线性的特征变化来来得到更加丰富的hypothesis class; 因此，尽管$\theta^Tx+\theta_0$是一个关于x的线性函数，但是$\theta^T\phi(x) + \theta_0$其实是一个关于x的非线性函数，如果$\phi$是一个关于x的非线性函数的话；也就是说，尽管我们现在讨论的回归问题是一个线性问题，但是我们也可以处理非线性问题，因为我们可以对输入值做一个映射，使得原来的非线性的值映射成线性的值。

> 我怎么感觉这个地方我理解的有点点问题啊。。。。

我们将会将回归问题看作是一个最优化问题，在这个最优化问题中，我们有数据集$\mathcal{D}$, 我们希望能够找到一个linear hypothesis 能够最小化均方误差；我们的目标函数，通常被称为是均方误差mse, 目标就是要找到一组参数$\Theta = (\theta, \theta_0)$, 能够最小化函数值：
$$
\mathcal{J} = \frac 1 n \sum_\limits{i=1}^n (\theta^Tx^{(i)} + \theta_0 - y^{i})^2
$$
我们要找的目标是：
$$
\theta^*, \theta_0^* = arg\min_{\theta, \theta_0}\mathcal{J}(\theta, \theta_0)
$$


### 1. Analytical solution: ordinary least squares

其实这里的均方误差 mean square error 是一个最小二乘的问题，这个问题是存在解析解的；在这一部分，我们就是要找到这个问题的解析解；在找linear hypothesis的时候，一个非常有意思的问题就是找到一个linear hypothesis,这个linear hypothesis 能够最小化均方误差(this general problem 通常被称为ordinary least squares OLS), 也就是所谓的普通最小二乘法；其实对于普通最小二乘，我们可以找到一个closed-formula 来解决这个问题；

> 到底什么是闭式解？genrally that it involves direct evaluation of a mathematical expression using a fixed number of "typical" operations(like arithmetic operations, trig functions, powers etc. )

Everyting is easier to deal with if we assume that $x^{(i)}$ have been augmented with an extra input dimension(feature) that always has value 1, so we may ignore $\theta_0$ 其实这个技巧在第2章和第3章我们就使用了，通过将其扩充到高维，将问题转化一下，我们就可以不需要考虑偏移量，这样一来可是使我们的分析变得简单；

我们会使用微积分的方法来求解这个最小二乘的问题，我们将目标函数$\mathcal{J}$对参数$\theta$进行求导，然后令其等于0，然后用来求解$\theta$的值；除了这些，我们还有另外一步需要做一下，因为我们只是求出了一介导数为零的点，可是我们并不知道这个点是极大值点还是极小值点，所以我们需要检查一下以确保这个点不是最大值点，也不是inflection point (拐点) ，但是我们并不会work through that here, 我们所需要做的操作只有：

- finding $\partial \mathcal{J}/\partial\theta_k$ for k in 1, …, d;
- Constructing a set of k equations of the form $\partial \mathcal{J} / \partial\theta_k = 0$, and 
- Solving the system for values of $\theta_k$

> 我们在这里表示所有的参数维数是$d$, 来表示特征的总数，在这里，d表示加了1的，也就是那个提高了一个维度的；

其实我们只需要做这三步就好了，to get parctice for applying techniques like this to more complex problems, we will work through a more compact (and cool ) matrix view. 下面就是用矩阵的观点来看这如何求最小二乘的解:

我们可以将我们的训练数据想象成矩阵$X$与矩阵$Y$, 对于矩阵$X$的每一列都是一个example, 这里我们仍然假设偏移量为0，特征的维度为$d$, 矩阵$Y$的每一列相应的对对应于一个target output value, 也就是所谓的label;
$$
X = 
\left[
\begin{matrix}
x_1^{(1)} & \cdots & x_1^{(n)} \\
\vdots & \ddots & \vdots \\
x_d^{(1)} & \cdots & x_d^{(n)} 
\end{matrix}
\right]
\ \ \ \ \ \ 
Y = 
\left[
\begin{matrix}
y^{(1)}  \cdots  y^{(n)} 
\end{matrix}
\right]
$$

> 可以看到$X$的维度为$d \times n$, Y的维度为$1 \times n$ 

在大多数的教材中，它们都认为一个单个的example $x^{(i)}$ 是一个行向量，而不是一个列向量，为了之后能哦股得到一个你能够认出来的一个解，我们将会定义一个新的矩阵和向量$W$与$T$, $W$与$T$ 分别是$X$与$Y$的转置:
$$
W = X^T = 
\left[
\begin{matrix}
x_1^{(1)} & \cdots & x_1^{(d)} \\
\vdots & \ddots & \vdots \\
x_d^{(1)} & \cdots & x_d^{(n)} 
\end{matrix}
\right]
\ \ \ \ \ \ 
T = Y^T = 
\left[
\begin{matrix}
y^{(1)}  \\
\vdots \\
y^{(n)} \\
\end{matrix}
\right]
$$

> 可以看到，这里W的shape 是$n \times d$, T的shape是$n \times 1$

我们知道$\theta$是我们要求的$d$维的列向量，这里用$W\theta - T$ 表示的就是一个$d \times 1$的列向量 这个列向量的每一个元素值都是预测值与真实值的差值，通过自己与自己做内积，然后求平均值，就可以得到我们的最小二乘的值；
$$
J(\theta) = \frac 1 n (W\theta - T)^T(W\theta-T) = \frac 1 n \sum\limits_{i=1}^n((\sum_{j=1}^d W_{ij}\theta_j) - T_i)^2
$$
然后我们使用矩阵/向量的微分，我们就可以得到:
$$
\nabla_\theta J = \frac 2 n W^T(W\theta-T)
$$

> 其实具体过程是这个样子的：
>
> 我们先对$\frac 1 n (W\theta - T)^T(W\theta-T)$, 求微分，其实矩阵的微分和多元函数微分在形式上是非常相像的，这里可以看作是两个部分的乘积，我们可以采用微分乘法的形式就可以得到:
> $$
> \frac 1 n ( \frac {\partial(W\theta - T)^T} {\partial \theta}(W\theta - T) + (W\theta - T)^T\frac {\partial(W\theta - T)}{\partial \theta}) \\
> = \frac 1 n (W^T(W\theta - T) + (W\theta - T)^TW)
> $$
> 通过观察，我们可以发现：$W^T$是一个$d \times n$， $(W\theta - T)$ 是一个shape 为$n \times 1$的列向量, 所以，$W^T(W\theta - T)$ 其实是一个$d \times 1$的列向量；$(W\theta - T)^T$ 是一个$1 \times n$的行向量，$W$是$n \times d$, 
>
> ### 等一下，这个地方没用弄明白！！！



我们让这个梯度等于0，那么我们就可以得到:
$$
\frac 2 n W^T(W\theta - T) = 0 \\
W^TW\theta - W^TT = 0 \\
W^TW\theta = W^TT \\
\theta = (W^TW)^{-1}W^TT
$$


> 在这里，我们进行这些变换都比较简单，但是有一点是值得注意的，那就是在最后一步，我们直接左乘了一个$(W^TW)^{-1}$, 因为$W$的shape 是$n\times d$的，$W^TW$的shape 是$d \times d$的，所以是一个方阵是没有问题的，再来考虑一下这个矩阵是不是可逆的？首先我们知道$W^TW$一定是一个对称矩阵，
>
> ### 这个地方也没有弄明白，把矩阵论的书拿过了翻一下！

我们可以来检查一下，看一看维数是不是正确的：
$$
\theta = (W^TW)^{-1}W^TT
$$
所以，通过这个闭式解，我们可以直接计算出能够最小化最小均方误差的参数$\theta$, 这个闭式解相当漂亮；



### 2 Regularization

然而，实际上，我们在自己实际操作的时候，可能会遇到一点麻烦，如果$W^TW$不可逆该怎么办？

> 比如说，有一种情况是数据集中只有两个点,这两个点的坐标是一样的$x^{(1)} = x^{(2)} = (1, 2)^T$, 那么我们的W就是矩阵：
> $$
> W = \left[
> \begin{matrix}
> 1 & 2 \\
> 1 & 2
> \end{matrix}
> \right]
> \ \ \ \ \ \ \
> W^T = \left[
> \begin{matrix}
> 1 & 1 \\
> 2 & 2
> \end{matrix}
> \right]
> \\
> W^TW = \left[
> \begin{matrix}
> 2 & 4 \\
> 4 & 8
> \end{matrix}
> \right]
> \ \ \ \ \ \ \
> (W^TW)^{-1} = ?
> $$
> 可以看到，不可逆了，因为$W^TW$的rank为1,并不是一个可逆矩阵

除了这个可逆矩阵的问题之外，另外一个问题就是关于过拟合的问题：我们建模了一个目标函数，这个目标函数的目标是尽可能的去拟合训练的数据，但是正如我们在"margin maximization"中强调过的，我们想要去regularize，也就是要进行一下正则化，以使得这个hypothesis不要太过于依赖训练集数据，也就是不要过拟合；

要处理上面的两个问题($W^TW$可能不可逆以及过拟合的问题)，我们引入一种称为`ridge regression`的技术；我们向`OLS`目标函数中加入正则项$||\theta||^2$。

> 一个思考问题: When we add a regularizer of the form $||\theta||^2$, what is our most "preferred" value of $\theta$, in the absence of data ? 
>
> 这个问题是什么意思啊？在没有数据的时候？在没有数据的时候是表示损失函数值为0吗？那么这个时候的目标函数就是$||\theta||^2$, 如果最小化这个目标函数的话，那么很显然就会使得$\theta$变为0

下面这个就是ridge regression 的目标函数:
$$
\mathcal{J_{ridge}}(\theta, \theta_0) = \frac 1 n \sum_{i=1}^n(\theta^Tx^{(i)} + \theta_0 - y^{(i)})^2 + \lambda||\theta||^2
$$
越大的$\lambda$会迫使$\theta$越来越靠近0；注意，我们不会乘法$\theta_0$, 我们的正则项中只会惩罚$\theta$,因为从直觉上来讲，$\theta_0$ is what "floats" the regression  surface to the right level for the data you have,  and **so you shouldn't make it harder to fit a data set where the y values tend to be aroud one million than one where they tend to be around one.** The other parameters controls the orientation of the regression surface, and we prefer it to have a not-too-crazy orientation:

要使得ridge loss 取得最小值，仍然是存在解析解的，但是相比于对OSL的求导，对于ridge的求导更加复杂，因为我们需要对$\theta_0$做特殊的处理，如果我们选择不对$\theta_0$做特殊的处理(我们的处理方式就像第2、3章一样，通过对输入的特征向量增加一个维度), 那么我们就可以得到ridge regression 目标函数的梯度；
$$
\nabla_\theta J_{ridge} = \frac 2 n W^T(W\theta-T) + 2\lambda\theta
$$
然后我们令这个梯度为0向量，并进行求解：
$$
\frac 2 n W^T(W\theta-T) + 2\lambda\theta = 0 \\
\frac 1 n W^TW\theta - \frac 1 n W^TT + \lambda\theta = 0 \\
\frac 1 n W^TW\theta + \lambda\theta = \frac 1 n W^TT \\
W^TW\theta + n\lambda\theta = W^TT \\
(W^TW + n\lambda I)\theta = W^TT \\
\theta = (W^TW + n\lambda I)^{-1}W^TT
$$
真实一个复杂的式子:
$$
\theta_{ridge} = (W^TW + n\lambda I)^{-1}W^TT
$$


> 这个回归就是所谓的"岭回归"， 之所以叫做岭回归，是因为我们在取逆矩阵之前，我们在$W^TW$上加上了一个$n\lambda I$ 这么一个对角矩阵；
>
> 为什么叫“岭”回归呢？这是因为按照岭回归的方法求取参数的解析解的时候，最后的表达式是在原来的基础上在求逆矩阵内部加上一个对角矩阵，就好像一条“岭”一样。加上这条岭以后，原来不可求逆的数据矩阵就可以求逆了**不过为什了加了个岭之后就可以求逆了呢？** 为什么当lambda 大于0的时候就是可逆的呢？
>
> 不仅仅如此，对角矩阵其实是由一个参数lamda和单位对角矩阵相乘组成。lamda越大，说明偏差就越大，原始数据对回归求取参数的作用就越小，当lamda取到一个合适的值，就能在一定意义上解决过拟合的问题：原先过拟合的特别大或者特别小的参数会被约束到正常甚至很小的值，但不会为零。
>
> 这里就解决了上面的线性回归的两个问题；
>
> 推导一下 ridge regression solution 的结果 Derive this version of the ridge regression solution



现在来讨论一下正则化的作用：

一般而言，在机器学习里，不仅仅是回归问题，正则化这个东西非常有用，在区分两种方式，这两种方式是作用在一个hypothesis $h \in \mathcal{H}$ 上，这两种方式可能会contribute to error 在训练集的数据上。我们有：

- Structural error: This is error arises because there is no hypothesis $h \in \mathcal{H}$ that will perform well on the data, for example because the data was really generated by a sin wave but we are trying to fit it with a line.
- Estimation error: This is error that arises because we do not have enough data(or the data are in some way unhelpful) to allow us to choose a good $h \in \mathcal{H}$

当我们增加$\lambda$的值的时候，我们往往会增加结构性误差，应为我们增大$\lambda$ 就是在增大正则化的权重，也就是说我们希望我们的模型不要太过于复杂，这样就有可能会导致模型缺乏拟合程度，从而导致结构性误差，但是增大$\lambda$的值会减小我们的估计误差，因为我们增大$\lambda$就是在增大正则项的权重，正则项会让我们拟合出来的模型对噪声不那么敏感，也就是说之后在测试集上对于误差/方差，estimation error 的影响就不会太大；

> 可以考虑一下，如果是应用一个polynomial basis of order k as a feature transformation $\phi$, on your data. Would increasing k tend to increase or decrease structural error ? What about estimation error ? 如果是用一个k阶的多项式特征变换函数的话，那么起到的作用和正则化刚好是相反的，刚好会减小结构性误差，增大estimation error(在测试集上的)
>
> 对于这两个概念：estimation error, 和 structural  error , 其实在机器学习中有两个专业的术语和它对应；这两个概念在机器学习更加高级的处理当中被深度的研究；我们一般将Structural error 称为bias, 将 estimation error 称为 variance , 一个是偏差，一个是方差；



### 3. Optimization via gradient descent

我们要求矩阵$W^TW$的逆矩阵，将会花费$O(d^3)$的时间复杂度，这个时间复杂度相当的高了，当矩阵的维度非常高的时候，这么高的计算复杂度使得求解析解变得非常的困难；如果我们有高维的数据，我们可以继续求助于梯度下降法；

> Why is having large n not as much of a computational problem as having large d?
>
> 因为$W^TW$的维度是$d \times d$， 这个和$n$并没有关系；

回忆一下岭回归的目标函数:
$$
\mathcal{J_{ridge}}(\theta, \theta_0) = \frac 1 n \sum_{i=1}^n(\theta^Tx^{(i)} + \theta_0 - y^{(i)})^2 + \lambda||\theta||^2
$$
岭回归相对于参数$\theta$的梯度为:
$$
\nabla_{\theta}J = \frac 2 n \sum_{i=1}^n(\theta^Tx^{(i)} + \theta_0 - y^{(i)})x^{(i)} +2 \lambda\theta
$$

> 这里有个地方要说明一下，对于二范数$||\theta||^2$求关于$\theta$的导数 
>
> 二范数： $\sqrt{x_1^2 + x_2^2 + …, + x_n^2}$  , 所以$||\theta||^2 = x_1^2 + x_2^2 + … + x_d^2 = \theta^T\theta$, 求导就是$2\theta$; 
>
> ### 其实对于矩阵求导数那一块好像还不是很熟！

岭回归对于$\theta_0$的偏导数为:
$$
\frac {\partial J}{\partial \theta_0} = \frac 2 n \sum_{i=1}^n (\theta^Tx^{(i)} + \theta_0 - y^{(i)})
$$
有了这些偏导数，我们就可以使用普通的梯度下降或者是随机梯度下降来进行求解了；

好就好在不管是OLS问题还是ridge regression 问题，这些问题都是凸的，这也就意味着只有一个最小值，这也就意味着只要步长足够的小梯度下降就会保证找到最优解；































