### 卷积神经网络

目前，我们已经学习了全连接神经网络，在这种全连接的神经网络中，一层的的神经元和下一层的神经元是全连接的。当我们对于从输入映射到输出一无所知到时候，采用这种全连接的方式是一种比较好的方式；我们将会要求网络去学习近似；但是，如果我们对我们正在研究的问题知道一些信息的时候呢？那么我们最好能够利用这些信息，构建出具有特点的神经网络结构；这样做的好处是可以节约计算资源，可以极大的减少训练所需要的数据量，并且可以得到一个泛化性和鲁棒的都比较好的解；

在最近几年，神经网络在信号处理领域取得到瞩目的成绩，处理的信号可能是空间上的(在二维上的照片，或者是三维 depth，或者是CAT扫描)也可能是时间域上的(比如所声音，或者是音乐)。如果我们知道目前我们解决的是一个信号处理的问题，我们就可以利用这个问题的invariant属性。在这一章节中，我们重点是关注二维空间上的问题，也就是处理图片的问题；但是我们也会用一维空间上的问题作为讲解的例子；在这之后，我们也会处理一些时间域上的问题。

想象一下，你要自己设计并且训练一个神经网络，这个神经网络的输入是一张图片，输出是对这张图片的分类；如果这张图片里面包含有一只猫，那么我们就将这张图片分类为正的，反之，将其分类为负的；一张图片被描述为一个二维的像素矩阵，每一个像素可能是由3个整数值来表示，这些整数值分别代表红绿蓝三种颜色代表的channels的强度；

存在两个非常重要的先验的结构知识，我们能够使用这些结构知识来解决这个问题：

- 空间位置：我们考虑通过这张图片中的像素点来找到一个猫，我们必须要考虑这个像素点的各种组合，因为这些像素点都是一个连着一个的，也就是说是一个挨着你一个，他们的空间位置是相对固定的；
- 位置转换不变性: 无论这只猫在这种图片中的哪个位置，描述一只猫的pattern of pixels 是相同的；无论这只猫在图片的哪个位置，我们都可以识别出这只猫；

> 这个 three-dimension depth 我不知道是啥；
>
> 信号处理的问题具有什么invariant的属性? 这个也没用弄懂；

接下来我们会设计一个神经网络结构，这个神经网络结构会用到上面这些性质。



### 1. Filters

我们首先从image filter 这个地方开始讨论，一个image filter是一个函数，这个函数takes in 一个局部的空间上的某个像素值的邻域(a local spatial neighborhood of pixel values), 然后检测出在那些数据上是否存在某些pattern(detects the presence of some pattern in that data.) 

我们首先来考虑一个非常简单的例子,在这个简单的例子中，我们的图片是一个一维的图片，这张图片的像素点的值要么为0，要么为1；我们让我们的 filter F 的大小为2，这个filter是一个只有2个数的向量，我们让这个filter沿着这张imgage移动，在每一个步长之下，我们让filter与image重合的那两个数做点乘，然后将这些点乘的结果聚集起来形成一张新的图片；

我们用$\bold X$表示原始的图片，这张图片的size是$d$, 然后，这张图片的输出我们表示成如下的式子:
$$
\bold Y_i = F^T(\bold X_{i-1}, \bold X_i)^T
$$
为了确保我们输出的图片的维度也是$d$, 我们一般会“pad” 输入的图片，with 0, 如果我们需要访问像素点，这些像素点都在输入图片的边界之外的话。这种将filter作用到图片，并且产生一张新的图片的过程，我们称为"卷积"

> 我之前都没注意，原来padding的过程是在输入的图片上做padding的啊?

下面是一个具体的例子. 我们让卷积核$F_1 = (-1, +1)$. 我们给出的第一张图片如下图所示：我们可以使用卷积核$F_1$对这张图片做卷积，来得到第二张图片，你可以将这个卷积核看作是一个检测器，这个检测器的作用是检测出原始图片的"left edges"， 要具体弄明白是什么回事，你可以看一看在output image 中值为1的点对应的image中的的值是什么值，我们可以看到，在input image中的值要么为0，要么为1，所以，两个点的可能组合为(00, 11, 10, 01), 这4种可能性与(-1, +1)做点乘，可能的值为(0, 0, -1, +1)，所以，如果输出的值为1的话，对应的在原图中的点就是0, 1, 也就是说这个filter (-1, +1)可以检测出(0, 1)也就是"left edges"; 另外一个比较有意思的filter就是$F_2 = (-1, +1, -1)$, 下面的第三种图片是使用卷积核$F_2$对第一张图片进行卷积操作得到的结果；这个卷积核可以被看作是用来检测一个单独的正像素的检测器,因为一张图片，如果是三个点可能的组合为(000, 100, 010, 001, 011, 101, 110, 111), 用$F_2$做卷积，得到的结果就是(0, -1, +1, -1, 0, -2, 0, -1), 可以看到，如果结果为+1的话，那么在原图中的输入必定是010， 所以，这个detector可以被看作是 isolated positive pixels in the binary image.

二维版本的卷积核被认为是在所有哺乳动物大脑的视觉皮层中发现的。类似的模式来自自然图像的统计分析。过去从事计算机视觉工作的人们常常花费大量时间在手工设计滤波器组， 滤波器组是一组滤波器(A filter bank is a set of sets of filters, arranged as shown in the diagram below.)，如下图所示排列。

所有在第一组中的filters都被应用到原始图片中，如果有k个这样的filters, 那么得到的结果就是k张新的图片这k张新的图片被称为是channels，现在，想象一下，我们将这k张图片堆叠起来，我们就会得到一个立方体的数据(Now imagine stacking all these new images up so that we have a cube of data)，怎么索引这些立方体中的数据呢？通过原图中的行和列，以及channel来进行索引；我们得到了这组立方体数据之后，我们接下来要继续使用filters,这个时候，这些filters通常都是三维的，每一个filter,each one will be applied to a sub-range of the row and column indices of the image and to all of the channels. 也就是说，这个三维的filter，会将这个三维的cube data给卷积完，通过图片的行，列以及channels这三个维度来完成；

在上面，我们所说的这个3维的数据块(These 3D chunks of data)，都被称为是张量tensor，关于张量的代数计算是非常有意思的，有点像矩阵的代数计算，在之后我们会使用流行的神经网络软件tensorflow来操作；

下面是应用一个二维卷积核的一个稍微复杂一点的例子。我们在第一层有两个$3 \times 3$的卷积核$f_1, f_2$, 你可以将卷积核$f_1$看成是找一个水平方向上具有连续三个的点，而$f_2$则是垂直方向量具有连续3个点的卷积核，假设我们输入的图形是$n\times n$的矩阵，那么经过这两个卷积核操作之后的结果就是一个$n \times n \times 2$的一个张量，现在我们应用一个张量卷积核，这个张量卷积核是3维的，这个tensor filter 是为了寻找一个combination of 两个水平的，和两个垂直的bar,  (now represented by individual pixels in the two channels), resulting in a single final n × n image.

当我们需要处理的是一张彩色图片的时候，我们就将这种图片看成是有3个channels, 因此相当于是一个$n \times n \times 3$的tensor;

接下来，我们就要设计具有这种结构的神经网络模型，Each “bank” of the filter bank 将对应着一个神经网络的一层；在每一个单独的filter中的数字都是神经网络中的权重，我们还是使用梯度下降 的方法来训练这些神经网络，有意思并且强大的是，相同的权重，在每一次计算中会被使用许多许多次；这个所谓的权重共享就意味着我们可以表达一个transformation，在一张非常大的图片上，用一些相对少的参数；这也意味着我们要小心的弄清楚怎么来训练这个神经网络

> 在这里为了表示的简单，我们假设所有的图片以及filters都是方形的

我们这样来定义一个filter layer $l$ formally with:

- number of filters $m^l$; 卷积核的个数

- size of filters $k^l \times k^l \times m^{l-1}$;  卷积核的大小

  > 可以看到，卷积核大小的第三个维度，一定是上一层的卷积核的数量

- stride $s^l$, 步长，is the spacing at which we apply the filter to the image ; so far  we have used a stride of 1, but if we were to “skip” and apply the filter only at odd-numbered indices of the image, then it would have a stride of two (and produce a resulting image of half the size); 

- input tensor size $n^{l-1} \times n^{l-1} \times m^{l-1}$, 输入的张量的大小；

  > 似乎这里并没有要求每次输出的图像的大小是一样的

这一层神经网络会输出一个张量为大小为 $n^{l} \times n^{l} \times m^{l}$, $n^l = \left \lfloor \frac{n^{l-1}}{s^l} \right \rfloor$, 权重的值就是在卷积核中定义的值输出的层会有$m^l$个不同的卷积核$k^l \times k^l \times m^{l-1}$， 所以，我们可以很容易的计算出在这一层中权重的值的个数:$m^l \times k^l \times k^l \times m^{l-1}$ 

> 这里之所以是$m^{l-1}$是因为卷积核做点乘是与上一层得到的结果对应做点乘的；

这个看起来可能有点复杂，但是，我们得到了丰富的映射，这些映射能够抽取出图片的结构，并且相比于全连接网络层，这种结构的权重更少；

在考虑一下上面的问题，如果使用的是全连接层而不是卷积层，那么考虑输入张量为



### 2. Max Pooling

我们将filter bank，结构化为一个金字塔型的形状是很合理的，在这个金字塔形状中，经过若干次连续的处理后，图片的形状会变得越来越小。

> 那为啥之前提出padding呢？padding 不是保持图片的形状不变的吗？

基本的想法是我们找到局部的模型 local patterns， like bits of edges in the early layers, and then look for patterns in those patterns, etc， 然后再再这些patterns 里面找其他的patterns。 这就意味着我们可以在一种非常大的图片里可以使用连续的使用filters来有效的查找patterns，如果我们移动的步长越大，那么最后得到的图片就会越小，但是不一定会聚合该空间范围内的信息。

另一种常见的神经网络层类型，它实现了这种信息的聚合， max pooling, 一个max pooling 层的作用就像一个filters一样，但是没有权重，你可以把它想象成是一个纯函数层，就像全连接层的ReLU激活层一样，max pooling 层又一个filter size，就想一个filter 层，但是只会返回它的field里面的最大的那个值，当具有以下特征的时候，我们就使用max pooling：

- 步长大于1，这样一来得到的图片的形状就比输入图片的形状的大小要小；
- $k \ge$ stride , 这样一来，整个图片都能够被覆盖到；

> 我没看懂他说的max pooling 实现信息的聚合是什么意思？Max pooling 不是相应的会丢掉一些信息的吗？

As a result of applying a max pooling layer, we don’t keep track of the precise location of a
pattern. 我们并没有跟踪一个pattern 的精确的位置，This helps our filters to learn to recognize patterns independent of their location.

Consider a max pooling layer of stride = k = 2. This would map a 64 × 64 × 3 image to
a 32 × 32 × 3 image.

> 有人想可以可以加两个max pooling 层, 当然也是size k， 其实 如果是这样的话，相当于是用一个stride = k = 4 Max pooling 层了；



### 3. Typical architecture

下面是一个典型的卷积神经网络的结构：

在每一个filter layer  卷积层之后，一般都有一个ReLU激活函数，可能会有多个filter/ReLU 层，然后有一个max pooling 层；then some more filter/ReLU layers, then max pooling. Once the output is down to a relatively small size, there is typically a last fully- connected layer, leading into an activation function such as softmax that produces the final output. The exact design of these structures is an art—there is not currently any clear theoretical (or even systematic empirical) understanding of how these various design choices affect overall performance of the network.

The critical point for us is that this is all just a big neural network, which takes an input and computes an output. The mapping is a differentiable function of the weights, which means we can adjust the weights to decrease the loss by performing gradient descent, and we can compute the relevant gradients using back-propagation!

> 实际上并不是可微分的，因为不仅仅是ReLU激活函数，而且还有 max pooling 操作都会导致不可微分；

我们来看一下一个非常简单的版本，来看一下反向传播是如何在一个卷积网络上工作的，cnn的架构如下，假设我们有一个一维的single-channel 的image,这个image 的大小为$n \times 1 \times 1$, 只有一个卷积核，这个卷积核的大小为$k \times 1 \times 1$, 在第一个卷积层上，Then we pass it through a ReLU layer and a fully-connected layer with no additional activation function on the output

为了简化，我们假设k是一个奇数，我们让输入的图片$\bold X = \bold A^0$, 我们假设我们使用的是均方误差，然后我们可以描述前向传播的过程:



> 对于一个filter 为k的卷积核，为了保证输出与输入是一样的大小，我们需要padding多少？因为输入输出的size 为 n, 如果不paddin ,那么输出的就是 n/k个，所以需要(n+d)/k = n，则 d = nk - n;





























































