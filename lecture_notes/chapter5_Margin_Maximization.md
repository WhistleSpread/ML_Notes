### 1 Machine learning as optimization

感知机算法最初被提出来完全是靠着作者的聪明才智与直觉，以及之后的理论分析才使得感知机算法得以完善。另外一种设计机器学习算法的方法是将机器学习算法看作是一个最优化的问题，然后使用标准的最优化算法并且设计并实施这些算法来找到hypothesis

我们首先写下一个目标函数$\mathcal{J(\Theta)}$, $\Theta$代表着模型中所有的参数。注意到又的时候我们也会写成$\mathcal{J(\theta, \theta_0)}$ 这是因为我们研究的是线性分类的问题我们的模型中的参数只有这两个；我们也经常这么来写$\mathcal{J(\Theta; D)}$, 这样写的目的是为了表明数据$\mathcal{D}$是与参数是依赖于数据集$\mathcal{D}$的；目标函数描述了我们对可能的假设参数$\mathcal{\Theta}$的感觉：我们通常是要找一组参数$\Theta$来最小化我们的目标函数值：
$$
\Theta^* = arg\min_\limits{\Theta}\mathcal{J(\Theta)}
$$

> 你可以把$\Theta^*$想像成是能够最小化目标函数$\mathcal{J}$的一组参数值

对于机器学习的目标函数来说，一个非常通用的形式是：
$$
\mathcal{J(\Theta)} = (\frac 1 n \sum_\limits{i=1}^n L(h(x^{(i)}; \Theta ), y^{(i)})) + \lambda \mathcal{R(\Theta)}
$$
前面的一部分是损失函数，后面的一部分是正则项，损失函数告诉我们的是我们对我们的预测$h(x^{(i)};\Theta)$，有多么的**不满意**。对于损失函数来说，一个比较常见的例子就是0-1损失函数，这个损失函数我们在第一章就已经介绍了：
$$
L_{01}(h(x;\Theta), y) = 
\begin{cases}
0 & if\ y= h(x;\Theta)\\
1 & otherwise
\end{cases}
$$
这个0-1损失函数告诉我们如果我们预测正确了，那么损失函数的值就是0，如果我们预测的不正确，那么损失值就是1，具体到线性分类器当中，这个0-1损失函数的具体表达就是:
$$
L_{01}(h(x;\theta, \theta_0), y) = 
\begin{cases}
0 & if\ y(\theta^Tx+\theta_0)>0\\
1 & otherwise
\end{cases}
$$


### 2 Regularization

如果我们所关心的只是找到一个hypothesis，这个hypothesis在训练的数据上具有比较小的损失值，那么我们就不需要进行正则化，那么在上面的机器学习的目标函数中就可以省掉第二部分的正则项了，但是，要记住，我们的目标函数是要在测试集的数据(也就是我们没有见过的数据)上表现的很好，你可能会认为这个是不可能的任务，但是人类和机器学习方法在这方面总是能做到比较出色；到底是什么能够让我们的机器学习算法对新的输入数据具有比较好的泛化能力的呢？其实就是潜在的正则项regularity在起作用，这个正则项即控制着训练数据，又控制着训练数据，又控制着测试数据。We have already discussed one way to describe an assumption about such a
regularity, which is by choosing a limited class of possible hypotheses. 另外一种方式就是提供一种更加平滑的guidance, 也就是说，在一个假设类中，我们偏好某些特定的假设类；我们通过常数$\lambda$来清楚的表达我们的这种偏好程度，这个常数$\lambda$就表达了我们有多愿意 trade off 在训练数据集上的损失值与在整个hypothesis 的变现效果(也就是泛化能力)；我们有多愿意牺牲一些在训练集上的损失值来换取更好的泛化能力；

这种trade off 可以用下面的图形很好的解释，假设的分隔平面$h_1$的训练误差为0， 但是这个模型相当的复杂，假设的分隔平面$h_2$尽管误分了两个点，但是这个分隔平面非常的简单，在不考虑解的请他belief的时候，我们通常希望我们的solution越简单越好，因此相比于$h_1$， 我们其实更加偏好于$h_2$, 因为我们遇见到$h_2$的表现将会比$h_1$更好；另外一种对正则化比较好的思维方式是我们不希望我们的hypothesis 太过于依赖具体的数据集：我们希望：如果训练集中的数据只是作出轻微的改变，那么hypothesis 并不会有很大的改变；

一个比较常用的确定的正则项就是:
$$
\mathcal{R(\Theta) = {||\Theta - \Theta_{prior}||}^2}
$$
我们在使用这个正则项的前提是我们对于参数$\theta$具有先验的知识，我们认为这些参数$\Theta$应该非常接近一些先验的参数值$\Theta_{prior}$, 如果我们没有这种对参数的先验的知识的话，那么我们就默认参数比较接近0，那么这个时候，正则项就变成了:
$$
\mathcal{R(\Theta) = {||\Theta||}^2}
$$


### 3 Maxmizing the margin

一个标准，我们用来判断一个分类器能够将所有的样例正确的分类(也就是我们的0-1损失函数值为0) ，我们说这样的一个标准就是要有比较大的margin, 回忆一下，一个标记点labeled point(x,y), 相应的超平面为$\theta \ \theta_0$ 那么这个点到separator的margin 就是:
$$
\gamma(x,y, \theta, \theta_0) = \frac {y(\theta^Tx+\theta_0)} {||\theta||}
$$
只有当一个点正确的分类了之后，这个margin 才是正的，而且margin的绝对值是一个点$x$到这个超平面的垂直距离，那么对于一个数据集来说，一个数据集对于一个超平面而言的margin就是这个数据集中的点到超平面的margin中的最小的那个margin，用数学公式来表达就是
$$
\min_\limits{(x^{(i)}, y^{(i)}) \in \mathcal{D}} \gamma(x^{(i)}, y^{(i)}, \theta, \theta_0)
$$
下面的一张图就是解释了两个不同的线性分类器有不同的margin, 我们可以看到，这两个线性分类器都能够将所有的样本点正确的分类，也就是0-1 loss 函数值都为0， 分类器$h_1$的margin 比分类器$h_2$的margin要大，从直觉上来讲，$h_1$具有更好的泛化性，能够对位置数据的分类表现的更好；相对而言，$h_2$的margin比较小，如果数据发生轻微的扰动的话，就可能会产生分类误差，所有泛化性相对于$h_1$而言没有那么好。

> 支持向量机在这门课上我们没有太多的时间来研究支持向量机，但是我们推荐你还是读一下支持向量机的内容

如果我们的数据是线性可分的，那么我们可以将这个分类问题转化为一个最优化的问题，我们的目标就是要找到一个具有最大margin 的分类器，也就是：
$$
\theta^*, \theta_0^* = arg\max_\limits{\theta, \theta_0}\min_\limits{i}\gamma(x^{(i)}, y^{(i)}, \theta, \theta_0)
$$
我们要找这样的一组参数，相当于是我们要最小化目标函数$\mathcal{J(\theta, \theta_0)}$
$$
\mathcal{J(\theta, \theta_0)} = -\min_\limits{i}\gamma(x^{(i)}, y^{(i)}, \theta, \theta_0)
$$
然而，这种目标函数形式用来进行优化的话会比较的麻烦，因为这个目标函数一次只对一个点敏感，因此我们不能对其进行求导，因此梯度下降的方法就不能适用了。

基于着出发点，我们就采用另外一种建模方式，我们首先假设我们可以猜到一个很好的目标值作为我们的margin, 我们暂时把它叫做$\gamma_{ref} > 0$, 然后我们就可以建立起模型了，我们想要找到一个分类器，这个分类器的margin 是$\gamma_{ref}$, 最终，我们的目标就是要找到一个值$\gamma_{ref}$以及一个分类器separator, 并且要满足下面两个条件:

- 所有点到separator的margin 都是要大于$\gamma_{ref}$的,并且
- 目标margin $\gamma_{ref}$ 不能太小，要比较大

对这两个条件说具体一点，我们首先定义一下新的损失函数,我们称这个损失函数为hinge loss

> 提醒一下，我们之前说损失函数有两类参数，一类是猜测的值，一类是实际的值，但是这个损失函数只有一类值，那就是margin， 一个margin 其实已经包含了两个参数了

$$
L_h(\gamma / \gamma_{ref}) = 
\begin{cases}
1-\gamma/\gamma_{ref} & if \ \gamma < \gamma_{ref} \\
0 & otherwise
\end{cases}
$$

下面的图形画出了一个hinge loss 函数，这个函数的自变量是margin $\gamma$, 如果margin $\gamma$ 比 $\gamma_{ref}$ 大，那么损失值就是0， 而且随着$\gamma$ 减小，损失值会增大，因为我们是希望margin $\gamma$ 比 $\gamma_{ref}$大的； 

在右边的图中，我们会解释一个超平面$\theta, \theta_0$, 平行于这个超平面，但是偏移量为$+\gamma_{ref}$或者是$-\gamma_{ref}$, 这分别是两个超平面，在图中是虚线，表示着我们的margin, 从图中可以看到，任何正确分类，但是在margin 之外的点的损失为0，任何正确分类，但是在margin 之内的点的损失在0到1之间，任何在没有正确分类的的损失大于等于1；

现在，我们会得到一个目标函数，关于参数$\Theta = (\theta, \theta_0, \gamma_{ref})$ 我们希望我们的margin_{ref} 越大越好，那么我们就在我们的目标函数中加入一个正则项也就是_$R(\theta, \theta_0, \gamma_{ref}) = \frac 1 {\gamma_{ref}^2}$ 这样一来，我们就有：
$$
\mathcal{J(\theta, \theta_0, \gamma_{ref}) = \frac 1 n \sum_{i=1}^n L_h(\frac {\gamma(x^{(i)}, y^{(i)}, \theta, \theta_0)} {\gamma_{ref}})} + \lambda(\frac 1 {{\gamma_{ref}^2}})
$$
我们可以看到，在目标函数中，这两项起的作用是相反的，前面的损失函数这一项是希望$\gamma_{ref}$越小越好，右边的一项$(\frac 1 {{\gamma_{ref}^2}})$，我们希望$\gamma_{ref}$越大越好；

> 关于这里的$\lambda$，在这里，因为lambda并不是hypothesis的参数，因为hypothesis的参数只有$\theta$和$\theta_0$, 但是这的$\lambda$是这个方法的参数，我们使用这个参数来选择hypothesis, 所有，我们将$\lambda$称为是超参数。
>
> 现在，你应该问一下你自己，如何选择超参数lambda, 因为对于超参数lambda 的选择不同，会导致最后的学习算法的不同；你认为应该如何选择？

现在，为了将我们的问题稍微做一下简化，并且与现存在更加标准的问题结合起来，我们需要采取一些unintuitive 的步骤，实际上，在我们的参数集合$(\theta, \theta_0, \gamma_{ref})$中，我们多了一个参数$\gamma_{ref}$, 最后要描述超平面的话，这个参数实际上是不必须的，所以我们在这里要做一点变形。

回忆一下，任何linear scaling 的$\theta$, $\theta_0$ 其实代表的是同一个separator， 因此我们同时放缩$\theta$与$\theta_0$的话，我们也不会分类器的表达以及margins的描述有任何的影响；我们可以同时缩放$\theta$和$\theta_0$使得：
$$
||\theta|| = \frac 1 {\gamma_{ref}}
$$

> 在这里看起来好想是我们直接把我们的目标margin $\gamma_{ref}$ 表示成了 $\frac 1 {||\theta||}$, 但是其实不是的，我们这样做是有我们的道理的，因为我们进行了缩放，但是这种缩放对于超平面并没有什么影响；

注意到，因为$\gamma_{ref}$与$||\theta||$是成反比例关系，我们希望$\gamma_{ref}$越大越好，那么反过来就是希望$||\theta||$越小越好；

使用了这个技巧之后，我们就不再需要$\gamma_{ref}$来作为我们的参数了，我们就可以得到下面的这个目标函数，这个目标函数其实就是支持向量机的函数，我们在之后也会经常提到这个函数；

在前面的第二章中我们参考了在第二章中的公式12和公式13:也就是一个点到一个超平面的margin, 和一个数据集相对于一个超平面的margin:
$$
y \cdot \frac {\theta^Tx + \theta_0} {||\theta||}
$$

$$
\min_\limits{i}(y^{(i)} \cdot \frac {\theta^Tx^{(i)} + \theta_0} {||\theta||})
$$

这样一来：
$$
\gamma = y \cdot \frac {\theta^Tx + \theta_0} {||\theta||} \\
\gamma_{ref} = \frac 1 {||\theta||}
$$
那么最后，公式12就转化成了:
$$
\mathcal{J(\theta, \theta_0) = \frac 1 n \sum_{i=1}^n L_h(y^{(i)}(\theta^Tx^i+\theta_0))} + \lambda{||\theta||}^2
$$
对于任何线性可分的数据集，下面有一些观察：

1. 如果$\lambda = 0$， 那么$\theta, \theta_0$总能够被选出来，从而能够让目标函数等于0

2. 如果$\lambda > 0$, 当时$\lambda$比较小，我们挑选$\theta$的标准是挑选出那个最小的$||\theta||$, 并且仍然保持很大程度上的数据可分性；

3. 如果$\lambda$非常大，那么表示我们可以忍受比较大的误差，因为我们更加偏好与一个形式更加简单(二范数)更小的分隔器separator

   > 一定要确保你可以详细的解释上面这三点，对于上面的第一点，如果我们要使目标函数接近于0，那么我们应该增大还是减小$\theta$的magnitude ? 要使得目标函数接近于0，就要尽可能的分对，所以对于正则项的权重要放低一点，所以应该增大$\theta$的magnitude; 对于上面的第三点，关于正则趋近于0，其实就是我们希望$\theta$的值有越多的趋近于0的越好；

在最优的情况下，对于线性可分的数据，并且有非常小的$\lambda$:

- $y^{(i)}(\theta^Tx^{(i)} + \theta_0) \ge 1$ 对于所有的$i$, 这样做的目的是为了保证对于所有的点的hinge loss 的值都是0
- $y^{(i)}(\theta^Tx^{(i)} + \theta_0) = 1$ 这种情况下的点至少要有1个，因为正则化这个过程会不断的迫使$||\theta||$尽可能的小，也就是说使得$\gamma_{ref}$尽可能的大， 当然这个是在保证这个超平面将所有的点都分类正确的前提下；

为了帮助你更好的理解，我们需要指出的是，对于那些在$y^{(i)}(\theta^Tx^{(i)} + \theta_0) = 1$上的点，这些点的margin实际上就是等于$\gamma_{ref}$,(否则的话，我们就会减少$||\theta||$以获得更大的margin, 对于这些lie along the margin 的点，我们使用这种有符号的距离 signed distance 来度量一个数据集到一个separator 的margin。
$$
(y^{(i)} \cdot \frac {\theta^Tx^{(i)} + \theta_0} {||\theta||}) = margin
$$

这里的$i$表示的点就是$y^{(i)}(\theta^Tx^{(i)} + \theta_0) = 1$ 上的点，所以就有：
$$
margin = \frac 1 {||\theta||}
$$

> 一定要注意公式19的成立条件： is only true under the assumptions listed at the top of this paragraph: when the data is separable, when λ is very small, and when θ is the optimum of the SVM objective. 











































