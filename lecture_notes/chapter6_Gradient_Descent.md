### Gradient Descent

在上一章节中，我们已经学到了如何来描述一个有意思的目标函数来描述一个机器学习模型，但是我们需要一种方法来找到最优的参数$\Theta^* = arg \min_\limits{\Theta}\mathcal{J(\Theta)}$. 其实在最优的的领域中，有许许多多的令人着迷的数学和算法基础，但是在本门课程中，我们将考虑最简单的一种方法，也就是**梯度下降法**

> 虽然是所有算法中最简单的，但是其实你应该考虑一下花几天功夫来学习一下这个算法

无论是在一维还是在二维空间中，从直觉上来讲，我们可以很容易的将$\mathcal{J(\Theta)}$看作是定义在$\Theta$上的一个平面，这个想法很显然可以扩展到高维理论当中，现在我们的目标就是要找到一组这样的参数$\Theta$, 从而使得这组参数$\Theta$对应的值是整个平面上的最小值；一种方式来思考梯度下降就是：你首先从平面上的任意点出发，look to see in which direction the “hill” goes down most steeply, take a small step in that direction, determine the direction of steepest descent from where you are, take another small step, etc.



### 1. One dimension

We will start by considering gradient descent in one dimension. Assume Θ ∈ R, and that we know both $J(Θ)$ and its first derivative with respect to$ Θ, J′(Θ)$. Here is pseudo-code for gradient descent on an arbitrary function f. Along with f and f′, we have to specify the initial value for parameter $Θ$, a *step-size* parameter $η$, and an *accuracy* parameter $ε$:

1D-GRADIENT-DESCENT($\Theta_{init}, \eta, f, f', \epsilon$)

$\Theta^{(0)} = \Theta_{init}$<br>t=0<br>**repeat**<br>	t = t + 1<br>	$\Theta^{(t)} = \Theta^{(t-1)} - \eta f'(\Theta^{(t-1)})$<br>**until** $|f'(\Theta^{(t)})| < \epsilon$<br>**return**$\Theta^{(t)}$

注意，除了上面所写的循环终止条件之外，还有其他的循环终止条件也是可以的，比如参数的magnitude的变化已经非常小了，或者是函数值的变化已经非常小了都是合理的$|\Theta^{(t)} - \Theta^{(t-1)}| < \epsilon$ 或者是 $|f(\Theta^{(t)}) - f(\Theta^{(t-1)})| < \epsilon$



**Th 1.1** 如果一个函数$\mathcal{J}$是凸函数， 对于任何比较合理精度的$\epsilon$, 存在某些步长$\eta$, 使得梯度下降会收敛到最优$\Theta$的$\epsilon$范围内；

> A function is convex if the line segment between any two points on the graph of the function lies above or on the graph.

然而，我们必须要小心，我们在选择下降的步长的时候，如果选择的步长太小，那么收敛的就会很慢，如果步长过长，那么就会发生振荡不收敛的情况However, we must be careful when choosing the step size to prevent slow convergence, oscillation around the minimum, or divergence.

If J is non-convex, where gradient descent converges to depends on θ*init*. When it reaches a value of θ where f′(θ) = 0 and f′′(θ) > 0, but it is not a minimum of the function, it is called a *local minimum* or *local optimum*.

> 对于凸函数来说，只有一个最小值点，极小值点就是最小值点；



### 2 Multiple dimensions

The extension to the case of multi-dimensional Θ is straightforward. Let’s assume$\Theta \in \mathcal{R^m}$,因此$\mathcal{J}: \mathcal{R} \rightarrow \mathcal{R}$, the gradient of $\mathcal{J}$ with respect to $\mathcal{\Theta}$ is：
$$
\nabla_{\Theta}J = 
\left[
\begin{matrix}
\partial J / \partial \Theta_1    \\
\vdots \\
\partial J / \partial \Theta_m   
\end{matrix}
\right]
$$
The algorithm remains the same, except that the update step in line 5 becomes
$$
\Theta^{(t)} = \Theta^{(t-1)} - \eta {\nabla_{\Theta}J}
$$
and we have to change the termination criterion. The easiest thing is to replace the test in line 6 with $|f(\Theta^{(t)} - f(\Theta^{t-1})) < \epsilon|$,  which is sensible no matter the dimensionality of Θ.



### 3 Application to SVM objective

There are two slight **“wrinkles”** involved in applying gradient descent to the SVM objective.

> 一个是对于矩阵的求导，一个是hinge loss 的微分的处理

我们在对支持向量机的目标函数使用随机梯度下降进行处理的时候，有两个小问题需要重点注意一下：我们首先来看一下svm的目标函数，以及我们需要做梯度下降所必须用到的梯度，在我们这个问题当中，我们考虑的是一个线性分类器，整个参数向量是由参数向量$\theta$以及一个标量$\theta_0$来表示的，因此我们需要对这些参数进行调整并且计算目标函数$\mathcal{J}$对它们分别求的偏导数；相应的目标函数和梯度如下公式所示：
$$
\mathcal{J(\theta, \theta_0) = \frac 1 n \sum_{i=1}^n L_h(y^{(i)}(\theta^Tx^i+\theta_0))} + \frac \lambda 2 {||\theta||}^2
$$

> 注意一下原来这里的$\lambda$变成了$\lambda / 2$，是为了之后的求导数方便；

$$
\nabla_{\Theta}J = \frac 1 n \sum_{i=1}^n L_h'(y^{(i)}(\theta^Tx^i+\theta_0))y^{(i)}x^{(i)} + \lambda \theta \\
\frac {\partial J}{\partial \theta_0} = \frac 1 n \sum_{i=1}^n L_h'(y^{(i)}(\theta^Tx^i+\theta_0))y^{(i)}
$$

注意一下，$\nabla_{\Theta}J$最后的结果应该是一个$d \times 1$的向量，而$\frac {\partial J}{\partial \theta_0}$ 将会是一个标量，因为在这里我们对$\theta$于$\theta_0$是分开求导数的; 假设参数$\theta$是$d \times 1$维的向量。

计算$\nabla_\theta{||\theta||}^2$, 我们定义这个向量的偏导数为$(\partial{||\theta||}^2/\partial{\theta_1}, …, \partial{||\theta||}^2/\partial{\theta_d})$，那么最后，$\nabla_\theta{||\theta||}^2$ 的维数也是$d \times 1$;我们要计算$\nabla_\theta(y(\theta^Tx+\theta_0))$ 我们定义的偏导数向量是$(\partial(y(\theta^Tx+\theta_0))/\partial\theta_1, …, \partial(y(\theta^Tx+\theta_0))/\partial\theta_d)$.

回忆一下hinge-loss:
$$
L_h(v) = 
\begin{cases}
1-v & if \ v < 1 \\
0 & otherwise 
\end{cases}
$$
这个损失函数不是可微分的，因为在$v = 1$ 的点导数不存在，因此我们考虑这个函数的次梯度：
$$
L_h'(v) = 
\begin{cases}
-1 & if \ v < 1 \\
0 & otherwise 
\end{cases}
$$
有了这个次梯度，对于$\nabla_\theta J$以及$\frac {\partial J} {\partial \theta_0}$ 的定义就完整了。最后，我们的梯度下降算法就变成了：

SVM-GRADIENT-DESCENT($\theta_{init},\theta_{0init}, \eta, J, \epsilon$)

$\theta^{(0)} = \theta_{init}$<br>$\theta_0 = \theta_{0init}$<br>t=0<br>**repeat**<br>	t = t + 1<br>	$\theta^{(t)} = \theta^{(t-1)} - \eta(\frac 1 n\sum_{i=1}^n \begin{cases} -1 & if \ y^{(i)} (\theta^{(t-1)T}x^{(i)} + \theta_0^{(t-1)}) < 1 \\ 0 & otherwise \end{cases}) y^{(i)}x^{(i)} + \lambda \theta^{(t-1)}$<br>	$\theta_0^{(t)} = \theta_0^{(t-1)} - \eta(\frac 1 n\sum_{i=1}^n \begin{cases} -1 & if \ y^{(i)} (\theta^{(t-1)T}x^{(i)} + \theta_0^{(t-1)}) < 1 \\ 0 & otherwise \end{cases}) y^{(i)}$<br>**until** $|f'(\Theta^{(t)})| < \epsilon$<br>**return**$\Theta^{(t)}$



### 4 Stochastic Gradient Descent

仍然是对于多维度的问题，当梯度的形式是一个和式的时候(就像上面的用梯度下降来求解svm的时候，这个时候梯度就是一个和式)，我们可以不同一次性考虑整个和式，我们可以随机的选择和式的某一项，然后在那个方向走一小步，这种做法看起来似乎有点疯狂，但是要记住，因为你最后要收敛到一个小地方，所以，如果你要留在一个地方，所有的小步骤将平均向与大步骤相同的方向发展。当然了你不会留在一个地方，所以你会移动，按照梯度期望的方向进行移动。

在机器学习中，大多数目标函数最后都会被写成是对数据点的求和，在这种情况下，使用随机梯度下降法就是从数据集中随机的选取一个数据点，然后来计算它的梯度，就好像数据集中只有这一个数据一样，然后朝着反方向走一小步；

我们假设我们的目标函数的形式如下：
$$
f(\Theta) = \sum_{i=1}^nf_i(\Theta)
$$
下面的伪代码是应该用SGD到一个目标函数中；



注意，我们现在并不固定步长$\eta$的值，这个步长是通过索引算法的迭代次数确定出来的，随着迭代次数的增加，SGD会收敛到一个局部最优值，所以，步长必须随着算法迭代次数的增多而不断的减小；

**Th 4.1** 如果目标函数$\mathcal{J}$是一个凸函数，并且步长变化的序列$\eta(t)$满足下面的条件:
$$
\sum_{t=1}^\infty \eta(t) = \infty \ \ and \ \  \sum_{t=1}^\infty \eta(t)^2 < \infty
$$
那么随机梯度算法几乎一定可以收敛到最优值$\Theta$.



下面是两个直觉，为什么随机梯度下降可能是一个比一般的梯度(批量梯度下降BGD)下降更好的选择：

- 跳出局部最优 If your f is actually non-convex, but has many shallow local optima that might trap
  BGD, then taking *samples* from the gradient at some point Θ might “bounce” you
  around the landscape and out of the local optima.
- 不会到达全局最优，这样就不会发生过拟合的现象 Sometimes, optimizing f really well is not what we want to do, because it might overfit the training set; so, in fact, although SGD might not get lower training error than BGD, it might result in lower test error.



































































































