---
layout:     post
title:      面向机器学习初学者的 MNIST 教程
subtitle:   千里之行始于足下
date:       2017-11-12
author:     YQY
header-img: img/post_tensorflow_2.jpg
catalog: true
tags:
    - Tensorflow
---

> 我是一个很懒的人，我想试试
>
> 希望我能坚持到最后，把tensorflow的官方教程全部翻译出来
>
> 提高自己，也帮助他人

# MNIST For ML Beginners

*本教程面向刚开始学习机器学习和 TensorFlow 的读者。如果你已经知道 MNIST 是什么，以及softmax(多项逻辑)回归是什么，那么你可能更适合这个[快速上手](https://tensorflow.google.cn/get_started/mnist/pros)教程。请在开始任何教程之前[安装好 TensorFlow](https://tensorflow.google.cn/install/)。*

当一个人开始学习编程的时候，一般来说第一件事就是打印"Hello World"。正如编程有 Hello World，机器学习有 MNIST 。

MNIST 是一个简单的机器视觉数据库。它是由类似下列的各种手写数字图像组成：

![img](https://tensorflow.google.cn/images/MNIST.png)

每张图像同时也包含了标签，告诉我们这个数字是多少。比如，上面的图像的标签分别是5，0，4，1。

在这个教程里面，我们将要训练一个模型用来查看图像并预测出上面的数字是多少。我们的目标不是训练一个真实精确，有着最高性能，尽管我们接下来将提供你代码来实现这个，而是简单使用 TensorFlow 。因此，我们将从一个非常简单的称为 Softmax Regression 模型开始。

本教程的实现代码非常的短，而且真正有意思的内容只发生在短短的三行中。然而，去理解代码后面的理念是非常重要的：TensorFlow 的工作原理和机器学习概念的核心。正因为如此，我们将通过这些代码小心的讲解这些。

## About this tutorial

本教程将逐行解释 [mnist_softmax.py](https://www.github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_softmax.py) 代码所做的事情。

你可以通过几种不同的方式来学习本教程，其中包括：

- 通过你阅读的每行代码的解释后，复制并粘贴每个代码片段到 Python 环境中。
- 在阅读理解之前和之后运行整个 `mnist_softmax.py` 文件，并使用本教程来理解你不清楚的代码部分。

我们将在本教程中完成以下任务：

- 学习关于 MNIST 数据集和 softmax regressions
- 根据查看图像中的每个像素来创建一个识别数字的模型
- 使用 TensorFlow 来 “看” 上千个数字图像例子来训练模型识别数字(并执行我们第一个 TensorFlow  会话来实现)
- 通过我们的测试数据来检测我们的模型的准确度

## The MNIST Data

MNIST 数据托管在 [Yann LeCun](http://yann.lecun.com/exdb/mnist/) 的网站上。如果你正在复制和粘贴本教程的代码，请先从这两行代码开始，它将自动下载并读取数据：

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

MNIST 数据集分为三个部分：55000 训练数据 (`mnist.train`)，10000 测试数据 (`mnist.test`)， 和 5000 验证数据 (`mnist.validation`)。这种划分是非常重要的：我们在还没有学习任何东西之前划分数据，这对于机器学习来说是非常有必要的。这样我们就可以确认我们的学习后的模型的实际泛化能力如何！

正如前面提到的，每个 MNIST 数据是由两部分组成的：一个手写数字图像和一个对应的标签。我们成图像为“x”，标签为“y”。训练集和测试集都包含了图像和对应的标签；例如训练图像是 `mnist.train.images` 而标签图像是 `mnist.train.labels` 。

每个图像是28x28像素组成。我们可以认为这个一个很大的数字数组：

![img](https://tensorflow.google.cn/images/MNIST-Matrix.png)

我们可以将这个28x28的行列式变成一个784的数组。只要我们保证每个图像采用相同的方式，那么我们并不在意如何把这个行列式展开。从这个角度来看，MNIST 图像集只是一个784维矢量空间的一束 [具有复杂结构的](https://colah.github.io/posts/2014-10-Visualizing-MNIST/) 点集而已(警告：可视化是计算密集型)。

展开这些数据将丢失图像的2D结构信息。这当然是不好的。最好的机器视觉方法会利用这种结构信息，我们将在之后的教程中讲解这些。但是在这里我们使用的简单模型，一个 softmax regressions(下面将会定义) 并没有使用这种信息。

结果是 `mnist.train.images` 是一个形状为 `[55000, 784]` 的张量(一个N维数组)。第一维是图像列表的一个索引，第二维是每个图像的每个像素的索引。张量中的每个条目是表示特定图像的特定像素的像素强度值，值介于0和1之间。

![img](https://tensorflow.google.cn/images/mnist-train-xs.png)

MNIST中的每个图像都有对应的一个标签数字，标签数字是0到9，代表了图像中画的数字。

为了本教程的目的，我们将要我们的标签作为 "one-hot vectors"。一个one-hot 向量是一个只有其中一个维度是1，其他维是0的向量。在这种情况下，向量上第n维的数字1表示第n的数字。例如3表示为[0,0,0,1,0,0,0,0,0,0]。所以 `mnist.train.labels` 是一个`[55000, 10]` 的数组。

![img](https://tensorflow.google.cn/images/mnist-train-ys.png)

现在我们准备好实际构造我们的模型了！

## Softmax Regressions

我们知道 MNIST 里的每个图像都是手写数字0到9中的一个。所以对于一张图像来说只有十种可能性。我们希望能够查看每张图并给出每张图的每个数字的概率。例如，对于我们的模型，在看一张手写数字9的图像，它能够80%确定这是9，但是有5%的可能是8(因为他们上半部分都是一个圆)，所有其他数字的可能性加起来是另外的15%，因为总共概率是100%。

这是 softmax regression 的一个简单，自然的经典模型。softmax 可以帮助你给几个不同的事物分别分配概率，因为 softmax 可以给我们一系列的0到1之间的值，并且总和达到1。之后当我们训练更复杂的模型的时候，最后一步也是要使用 softmax 层。

softmax regression 有两个步骤：首先我们将我们输入的证据(evidence) 加入到确定的类别中。然后我们计算对应的证据的概率。

为了收集给定图像在特定类别上的证据，我们进行了像素值的加权求和。如果图像在某个类上有很高的像素强度的证据则权重为正，相反则为负。

下图显示了一个模型在学习每个类别的权重，红色表示负权重，而蓝色表示正权重。

![img](https://tensorflow.google.cn/images/softmax-weights.png)

我们也另外添加了一些被称为偏差的证据。基本上，我们希望能够有些东西独立于输入。结果对于给定输入 $x$ 属于特定类 $i$ 的证据可以表示为：
$$
\text{evidence}_i = \sum_j W_{i,~ j} x_j + b_i
$$
其中，$W_i$ 表示权重，$b_i$ 表示类$i$的偏置，$j$ 表示输入图像 $x$ 的所有像素的索引。然后我们用 softmax 函数将这些证据计算出概率 $y$ ：
$$
y = \text{softmax}(\text{evidence})
$$
这里的 softmax 被视为是一个激励(activation)或者链接(link)函数，在这种情况下，将我们的线性函数的输出塑造成我们想要的形式——一种10个数字分类的概率分布。你可以认为是将我们的输入的证据转化为我们在每个分类上的概率，定义如下：
$$
\text{softmax}(evidence) = \text{normalize}(\exp(evidence))
$$
展开等式：
$$
\text{softmax}(evidence)_i = \frac{\exp(evidence_i)}{\sum_j \exp(evidence_j)}
$$
但是把 softmax 视为激励是有利的：将输入值指数化，然后正则化。指数意味着每多一个单位的证据增加了假设(hypothesis)中乘数的权重。反之，每少一个单位的证据意味着减少了假设中乘数的权重。假设是不会有零或者负权重的。然后 softmax 归一化权重，使得总权重为一，形成一个有效的概率分布。(更多有关于softmax函数的信息请查阅 Michael Nielsen 书籍的有关[部分](http://neuralnetworksanddeeplearning.com/chap3.html#softmax)，其中包含交互式可视化的内容。)

你可以将 softmax 回归看成如下图所示，有很多的输入 $x$。对于每个输出，我们计算 $x$ 的加权求和，加上偏置，然后在应用于 softmax 函数中。

![img](https://tensorflow.google.cn/images/softmax-regression-scalargraph.png)

写成等式，如下：

![img](https://tensorflow.google.cn/images/softmax-regression-scalarequation.png)

我们可以将这个过程“矢量化”，转化为矩阵乘法和向量加法。这有助于提高计算效率。(也是一个更有用的思考方式。)

![img](https://tensorflow.google.cn/images/softmax-regression-vectorequation.png)

更简洁的写法如下：
$$
y = \text{softmax}(Wx + b)
$$
现在让我们将其用 TensorFlow 表示。

## Implementing the Regression

为了使用 Python 进行高效的数值计算，我们通常使用诸如 [NumPy](http://www.numpy.org/)这样的的库来做复杂的操作，例如矩阵乘法，它使用另一种高效的语言来实现。不幸的是，每一次运算的结果都会返回给 Python，这会带来许多额外的开销。如果你希望在GPU或者以分布式计算的方式来运行的话，这其中的转化数据的成本将非常高，带来的开销将更加糟糕。

TensorFlow 也在 Python 之外进行繁重的计算，但是为了避免这种开销，TensorFlow 做了更进一步的完善。TensorFlow 不在 Python 单独运行一个昂贵的操作，而是让我们描述一整个交互操作的图，然后再一起在 Python之外运行。(这种类似的情况可以在其他的机器学习库里看见。)

为了使用TensorFlow，首先我们需要载入它。

```python
import tensorflow as tf
```

我们通过操作符号变量来描述这些交互操作。现在让我们来创建一个：

```python
x = tf.placeholder(tf.float32, [None, 784])
```

`x` 不是一个特定的值。它是一个占位符(`placeholder`)，当我们让 TensorFlow 执行计算的时候让我们输入这个值。我们希望能够输入任意数量的 MNIST 图像，将其展开为一个784维的矢量。我们用一个2维浮点数张量来表示这些图，其形如 `[None, 784]`。(这里`None` 表示这个维度可以是任意长度的。)

对于我们的模型，我们一样需要权重和偏置。我们也可以设想把他们当作额外的输入，但是 TensorFlow 有一个更好的方式来处理它：`Variable`。一个 `Variable` 是一个可变的张量，存在于 TensorFlow 的可交互的图中。它可以在计算中被使用和被修改。对于机器学习应用来说，通常将模型参数设置为 `Variable`。

```python
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```

我们通过`tf.Variable`给`Variable`初始值来创建这些 `Variable`：在这里，我们用全为零的张量来初始化`W`和`b`。既然我们要学习`W` and `b`，那么他们被初始化成什么并不重要。

注意，`W`形如[784, 10]，因为我们希望乘以784维的图像矢量来产生一个10维的不同类的证据的向量。`b`的形状是[10]，所以我们可以把它加在输出上。

现在我们可以实现我们的模型了。只需要一行代码来定义它！

```python
y = tf.nn.softmax(tf.matmul(x, W) + b)
```

首先，我们通过`tf.matmul(x, W)`来使`W`乘以`x`。对应方程中的 $Wx$ ，这里 $Wx$ 作为一个小技巧来处理多项输入的2维张量`x`。然后我们加上`b`，最后应用`tf.nn.softmax`。

至此，经过两行代码设置变量后，我们只使用了一行代码来定义我们的模型。这不是因为 TensorFlow 是被设计为使用 softmax 回归特别简单：而这是一个非常灵活的方式来描述许多类型的数值计算，从机器学习模型到物理模拟仿真都是如此。而一旦定义后，我们的模型可以运行在不同的设备中：你的电脑的CPU上，GPU上，甚至是手机上！

## Training

为了训练我们的模型，我们需要定义什么样的模型是好的。那么实际上，在机器学习中我们通常定义什么样的模型是坏的。我们称之为代价(cost)，或者是损失(loss)，这代表着我们的模型与我们的期望的差距有多远。我们尝试使这个误差最小，因为误差越小，模型也就越好。

一个非常常见，非常好的损失模型被称为交叉熵("cross-entropy")。交叉熵源于信息理论里的信息压缩编码思想，现在在很多领域中其演变为一种重要的思想，从博弈论到机器学习都是如此，它被定义为：
$$
H_{y'}(y) = -\sum_i y'_i \log(y_i)
$$
其中$y$是我们的预测概率分布，$y'$是实际分布(数字标签的one-hot向量)。更粗糙的说话是，交叉熵是用于衡量我们的预测描述的实际的低效性。理解交叉熵更多的细节是超出本教程的范围的，但是它是值得[理解](https://colah.github.io/posts/2015-09-Visual-Information)的。

为了实现交叉熵，我们首先增加一个占位符用来输入正确的答案：

```python
y_ = tf.placeholder(tf.float32, [None, 10])
```

然后我们可以实现交叉熵函数，$-\sum y'\log(y)$：

```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```

首先， `tf.log` 计算了`y`的每个元素的对数。然后我们将`y_`与相对于的`tf.log(y)`相乘。再然后由于`reduction_indices=[1]` ，`tf.reduce_sum` 将y的第二维元素求和。最后，`tf.reduce_mean` 计算批处理中所有样本的平均值。

注意，在源代码中，我们不适用这个公式，因为它在数值上是不稳定的。相反，我们应用`tf.nn.softmax_cross_entropy_with_logits`在非归一化的logits上( 例如，我们在`tf.matmul(x, W) + b`中调用`softmax_cross_entropy_with_logits` )，因为在softmax激活中的数值计算是更稳定的。在你的代码中，请考虑使用`tf.nn.softmax_cross_entropy_with_logits`替代。

现在我们知道我们想要我们的模型做什么了，对于 TensorFlow 来说训练它来做这些是非常简单的。因为 TensorFlow 知道你所有计算的图，它可以自动使用[反向传播算法](https://colah.github.io/posts/2015-08-Backprop)来有效的确定你的变量如何影响你要求最小化的损失。而且它可以应用你选择的优化算法来修改变量并减小损失。

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

在这种情况下，我们让 TensorFlow 使用学习率为0.5的[梯度下降](https://en.wikipedia.org/wiki/Gradient_descent)(gradient descent algorithm)来最小化`cross_entropy`。梯度下降是一种简单的过程，其中 TensorFlow 将每个变量往降低成本的方向微小的变化。但是 TensorFlow 也提供了[许多其他的优化算法](https://tensorflow.google.cn/api_guides/python/train#Optimizers)：使用它是简单的如同调整一条直线一样。

TensorFlow 在这里实际上所做的是，在后台中实现反向传播和梯度下降的地方增加一个新的操作到你的图里。然后它返回给你一个单独的操作，当运行时，进行梯度下降训练，微调你的变量使得损失减小。

现在我们可以在`InteractiveSession`中启动模型：

```python
sess = tf.InteractiveSession()
```

首先我们创建一个操作来初始化我们创建的变量：

```
tf.global_variables_initializer().run()

```

开始训练——我们执行循环训练1000次：

```
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

```

在循环的每次中，我们“批量”从训练数据集中随机抽取100个数据。我们用批数据替换`占位符`来运行`train_step` 。

使用小批量随机数据称为随机训练，在这种情况下称为随机梯度下降。理想情况下，我们希望在每一步的训练中使用所有的数据，因为这样我们可以更好的了解我们应该做什么，但是这样做将带来很大的开销。因此，在每一步的训练中我们每次使用不同的子集。这样的开销不会太大而且可以得到相同的好处。

## Evaluating Our Model

那么我们的模型性能如何？

首先，让我们找出那些我们预测正确的标签。 `tf.argmax` 是一个非常有用的函数，它能给你一个张量中某个维度上最大值的索引。例如， `tf.argmax(y,1)` 是 我们模型认为每个输入最有可能的标签，其中`tf.argmax(y_,1)` 是正确的标签。我们可以使用 `tf.equal` 来检验我们的预测与真实是否匹配。

```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

这段代码会给我们一组布尔值。为了确定正确的比例，我们将其转化为浮点数。例如，`[True, False, True, True]` 变成 `[1,0,1,1]` ，这样其正确率为0.75。

```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

最后，我们计算我们模型在测试集上的准确性。

```python
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

这大概为92%。

这个结果很好吗？不怎么样。实际上这个结果很差。这是因为我们使用了一个非常简单的模型。做了一些小的调整后，我们将得到97%的准确率。最好的模型能得到大于99.7%的准确率！(想了解更多信息，请查看这个[不同结果的列表](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results) 。)

重要的是我们从这个模型中所学到的东西。不过如果你对这里的结论感到一点失望，请查看 [下一个教程](https://tensorflow.google.cn/get_started/mnist/pros) ，我们使用 TensorFlow 构建一个更复杂的模型已得到更好的结果！
