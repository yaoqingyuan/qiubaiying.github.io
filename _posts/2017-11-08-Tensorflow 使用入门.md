---
layout:     post
title:      TensorFlow 使用入门
subtitle:   千里之行始于足下
date:       2017-11-08
author:     QY
header-img: img/post_tensorflow_1.jpg
catalog: true
tags:
    - Tensorflow
---

> 我是一个很懒的人，我想试试
>
> 希望我能坚持到最后，把tensorflow的官方教程全部翻译出来
>
> 提高自己，也帮助他人

# Tensorflow 入门

本篇是Tensorflow编程入门指南。在开始本教程前，[安装 TensorFlow](https://tensorflow.google.cn/install/index)。为了充分利用本指南，你需要知道以下知识：

*   如何 Python 编程。
*   至少知道一点 Arrays 知识。
*   最好能懂一点机器学习的知识。即使，你一点机器学习的知识都不懂也没有关系，这依然可以是你阅读的第一份指南。

TensorFlow 提供了多个 APIs。最低级别的 API--TensorFlow Core-- 提供完整的编程控制。我们建议TensorFlow Core 用于机器学习研究者和其他需要对模型进行精确控制的人员。更高级别的 APIs 建立在 TensorFlow Core之上。这些更高级别的 APIs 相比于 TensorFlow Core 也更加容易学习与使用。另外，更高级别的 API 使得不同用户之间的重复任务更容易和一致。像 tf.estimator 这样的高级 API 可以帮助您管理数据集，评价，培训和推理。
本指南以 Tensorflow Core 开始。接下来，我们将演示如何在 tf.estimator 中运行相同的模型。理解 Tensorflow Core原理将在你使用更紧凑高级别 API 时能够更好的理解事情内部的工作原理。

# Tensors

Tensorflow 数据的核心单位是张量 **(tensor)**。一个张量是由一组任意维度的原始值组成。张量的 **(rank)** 是它的维数, 下面是一些张量的例子:
```python
3  # a rank 0 tensor; a scalar with shape []  
[1.,  2.,  3.]  # a rank 1 tensor; a vector with shape [3]  
[[1.,  2.,  3.],  [4.,  5.,  6.]]  # a rank 2 tensor; a matrix with shape [2, 3]  
[[[1.,  2.,  3.]],  [[7.,  8.,  9.]]]  # a rank 3 tensor with shape [2, 1, 3]
```

## TensorFlow Core 教程

### Importing TensorFlow

TensorFlow 程序规范导入语句如下:

```python
import tensorflow as tf
```

这使得 Python 可以访问所有的 TensorFlow 类(classes), 方法(methods)和符号(symbols)。接下来大多数的教程都假设你已完成了此项工作。

### The Computational Graph

你可能会认为 Tensorflow Core 程序都是包含以下两个部分：

1.  建立 computational graph。
2.  运行 computational graph。

计算图 **(computational graph)** 是一系列排列成节点图的 TensorFlow 操作。让我们来构建一个简单的计算图。每个节点将输入零个或多个张量并产生一个张量作为输出。每个类型的节点都是一个常量。类似于所有的 TensorFlow 常量，它没有输入，而只输出一个内部存储值。我们可以创建两个浮点型张量`node1` 和 `node2`：

```python
node1 = tf.constant(3.0, dtype=tf.float32) 
node2 = tf.constant(4.0)  # also tf.float32 implicitly  
print(node1, node2)
```

最后打印如下：

```python
Tensor("Const:0", shape=(), dtype=float32)  Tensor("Const_1:0", shape=(), dtype=float32)
```

注意，打印节点不会像你预期的那样输出 `3.0` 和`4.0` 。相反，他们在评估时将会分别输出 3.0 和 4.0. 为了实际评估节点，我们必须在 **session** 内运行计算图。session 封装了 Tensorflow 运行时的控制和状态。

以下代码创建了一个 `Session` 对象并调用了 `run` 方法运行计算图来评估`node1` 和 `node2`。通过在 session 中运行计算图：

```python
sess = tf.Session()  
print(sess.run([node1, node2]))
```

我们就可以看到期望的答案 3.0 和 4.0:

```python
[3.0,  4.0]
```

我们可以通过将张量节点与操作相结合来构建更复杂的计算(操作也是节点)。例如，我们可以增加两个常量节点并产生一个新的图形(graph):

```python
from future import print_function
node3 = tf.add(node1, node2)  
print("node3:", node3)  
print("sess.run(node3):", sess.run(node3))
```

最后两个打印语句：

```python
node3: Tensor("Add:0", shape=(), dtype=float32) 
sess.run(node3):  7.0
```

 

Tensorflow 提供一个名为TensorBoard的有效的程序，它可以将计算图显示为图形。以下一个截屏显示了TensorBoard将图形可视化：

![TensorBoard screenshot](https://tensorflow.google.cn/images/getting_started_add.png)

就目前来看，这张图并不特别有趣，因为它总是产生一个不变的结果。图形可以被参数化来接收称为占位符的外部输入。占位符是接下来提供值的变量。

````python
a = tf.placeholder(tf.float32) 
b = tf.placeholder(tf.float32) 
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
````

 这三行有点像一个函数(function)或者一个lambda，其中我们定义了两个参数(a和b)和一个操作。我们可以通过使用 [run 方法](https://tensorflow.google.cn/api_docs/python/tf/Session#run) 的 feed_dict 参数将多个输入的具体值提供给占位符来评估这个图形：

```python
print(sess.run(adder_node,  {a:  3, b:  4.5}))  
print(sess.run(adder_node,  {a:  [1,  3], b:  [2,  4]}))
```

输出结果：

```python
7.5  
[  3. 7.]
```

在 TensorBoard，这图形看起来是这样的：

![TensorBoard screenshot](https://tensorflow.google.cn/images/getting_started_adder.png)

我们可以通过添加其他的操作来使得计算图形更加的复杂，例如，

```python
add_and_triple = adder_node *  3.  
print(sess.run(add_and_triple,  {a:  3, b:  4.5}))
```

 输出结果：

```	python
22.5
```

在TensorBoard中，上面的计算图如下所示：

![TensorBoard screenshot](https://tensorflow.google.cn/images/getting_started_triple.png)

在机器学习中，我们通常需要一个可以计算任意输入的模型，比如上面的模型。为了使得模型可训练，我们需要能够修改图形使得其具有相同的输入能得到新的输出。变量(**Variables**)允许我们添加可训练的参数到图形中。他们可以被构造成一个类型和初始值：

```python
W = tf.Variable([.3], dtype=tf.float32) 
b = tf.Variable([-.3], dtype=tf.float32) 
x = tf.placeholder(tf.float32) 
linear_model = W*x + b
```

 当你调用`tf.constant` 时，常量就被初始化了，并且不能被修改。相比之下，当你调用`tf.constant` 的时候变量并没有被初始化。为了初始化TensorFlow里面的变量，你必须显式的调用下面的操作：

```python
init = tf.global_variables_initializer() 
sess.run(init)
```

实现`init` 是TensorFlow子图初始化所有的全局变量的一个句柄是很重要的。在我们调用`sess.run` 之前，变量是未初始化的。

由于`x` 是一个占位符，我们可以同时为多个`x` 评估`linear_model` :

```python
print(sess.run(linear_model,  {x:  [1,  2,  3,  4]}))
```

输出结果：

``` python
[  0. 0.30000001 0.60000002 0.90000004]
```

我们创建了一个模型，但是我们不知道这个模型的好坏。为了评估训练数据的模型，我们需要一个`y` 占位符来提供期望值，并且我们需要写一个代价函数。

代价函数用于衡量当前模型与预期值之间的差距有多远。我们将使用现行回归的标准代价模型，其中代价是当前模型与期望值之间的差距的平方的总和。`linear_model - y` 创建一个每个元素与其对应实例的误差的向量。我们调用`tf.square` 来计算这个误差的平方。然后我们统计所有的平方差来创建一个标量，通过调用`tf.reduce_sum` 来抽象所有例子的错误：

```python
y = tf.placeholder(tf.float32) 
squared_deltas = tf.square(linear_model - y) 
loss = tf.reduce_sum(squared_deltas)  
print(sess.run(loss,  {x:  [1,  2,  3,  4], y:  [0,  -1,  -2,  -3]}))
```

输出结果：

```python
23.66
```

我们可以通过重新将`W` 和`b` 分别赋值为-1和1来手动改进。一个变量可以通过`tf.Variable` 来初始化，也可以使用像`tf.assign` 的操作来改变。例如，`W=-1` 和 `b=1` 就是我们这个模型的最优参数，我们可以对应改变 `W` 和`b` 的值：

```python
fixW = tf.assign(W,  [-1.]) 
fixb = tf.assign(b,  [1.]) 
sess.run([fixW, fixb])  
print(sess.run(loss,  {x:  [1,  2,  3,  4], y:  [0,  -1,  -2,  -3]}))
```

打印的损失值现在为零。

```python
0.0
```

 我们能够猜测`W` 和 `b` 的最佳值，但是机器学习的重点是自动找到正确的模型参数。我们将在下一节展示如何完成这项工作。

## tf.train API

对于机器学习的完整讨论已超出了本教程的范围。但是Tensorflow提供的优化器**(optimizers)** 可以逐步改变每个变量来逐步使得代价函数达到最小值。最简单的优化器是梯度下降**(gradient descent)** 。它根据相关变量的倒是的减小的大小来修改每一个变量值。一般来说，手动计算符号导数是繁琐而又容易出错的。所以只要使用`tf.gradients` 来描述模型tensorflow就可以自动计算导数。简单来说，优化器通常会自动为你做这件事，例如：

```python
optimizer = tf.train.GradientDescentOptimizer(0.01) 
train = optimizer.minimize(loss)


sess.run(init)  # reset values to incorrect defaults.  
for i in range(1000): 
  sess.run(train,  {x:  [1,  2,  3,  4], y:  [0,  -1,  -2,  -3]})  
print(sess.run([W, b]))
```

最终的模型参数结果如下：

```python
[array([-0.9999969], dtype=float32), array([  0.99999082], dtype=float32)]
```

现在我们已经完成了一个实际的机器学习了！虽然这个简单的线性模型不需要太多的tensorflow core代码，但是使用更复杂的模型和方法给我们的模型提供数据则需要更多的代码。因此tensorflow为常见的模型，结构和功能提供更高级别的抽象。我们将在下一节中学习如何使用这些抽象。

### Complete program

完成可训练的现行回归模型如下所示：

```python
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```

运行后结果如下：

```python
W:  [-0.9999969] b:  [  0.99999082] loss:  5.69997e-11
```

 注意，loss是一个非常小的数字(非常接近于零)。如果你运行这段代码，你的loss可能和上面的loss不完全一致，因为模型的初始化是使用伪随机值的。

在tensorboard仍然可以可视化非常复杂的程序。![TensorBoard final model visualization](https://tensorflow.google.cn/images/getting_started_final.png)

## `tf.estimator`

`tf.estimator` 是一个高级别的 TensorFlow 库，它简化了机器学习的机制：

*   运行训练循环
*   运行评估循环
*   管理数据集

tf.estimator 定义了许多常见的模型。

### Basic usage

注意在 `tf.estimator` 里，线性回归将变得多么的简单:

``` python
# NumPy is often used to load, manipulate and preprocess data.  
import numpy as np 
import tensorflow as tf

# Declare list of features. We only have one numeric feature. There are many  
# other types of columns that are more complicated and useful. 
feature_columns =  [tf.feature_column.numeric_column("x", shape=[1])]  
# An estimator is the front end to invoke training (fitting) and evaluation  
# (inference). There are many predefined types like linear regression,  
# linear classification, and many neural network classifiers and regressors.  
# The following code provides an estimator that does linear regression. 
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)  

# TensorFlow provides many helper methods to read and set up data sets.  
# Here we use two data sets: one for training and one for evaluation  
# We have to tell the function how many batches  
# of data (num_epochs) we want and how big each batch should be. 
x_train = np.array([1.,  2.,  3.,  4.]) 
y_train = np.array([0.,  -1.,  -2.,  -3.]) 
x_eval = np.array([2.,  5.,  8.,  1.]) 
y_eval = np.array([-1.01,  -4.1,  -7,  0.]) 
input_fn = tf.estimator.inputs.numpy_input_fn(  
	{"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True) 
train_input_fn = tf.estimator.inputs.numpy_input_fn( 
	{"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False) 
eval_input_fn = tf.estimator.inputs.numpy_input_fn(  
	{"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)  
	
# We can invoke 1000 training steps by invoking the  method and passing the  
# training data set. 
estimator.train(input_fn=input_fn, steps=1000)  

# Here we evaluate how well our model did. 
train_metrics = estimator.evaluate(input_fn=train_input_fn) 
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)  
print("train metrics: %r"% train_metrics)  
print("eval metrics: %r"% eval_metrics)
```

运行后，结果类似如下：

```python
train metrics:  {'average_loss':  1.4833182e-08,  'global_step':  1000,  'loss':  5.9332727e-08}  eval metrics:  {'average_loss':  0.0025353201,  'global_step':  1000,  'loss':  0.01014128}
```

注意，不论我们评估的loss有多高，但它仍然接近与零。这意味这我们的学习是正确的。

### A custom model

`tf.estimator` 不会限制你在预定义模型中。假设我们想要创建一个自定义的 TensorFlow 没有的模型。我们仍然可以使用 `tf.estimator` 来使用高度抽象的数据集，处理，训练。为了说明，我们将展示如何利用我们的知识使用低级别的 TensorFlow API 来实现相同的现行回归模型。

为了定义一个使用`tf.estimator` 的自定义模型，我们需要使用 `tf.estimator.Estimator` 。实际上 `tf.estimator.LinearRegressor` 是 `tf.estimator.Estimator` 的一个子类。为了替代子类`Estimator` ，我们只需要提供 `Estimator` 的一个函数 `model_fn` 来告诉 `tf.estimator` 如何评估预测，训练步骤和损失。代码如下：

```python
import numpy as np 
import tensorflow as tf 

# Declare list of features, we only have one real-valued feature  
def model_fn(features, labels, mode):  
	# Build a linear model and predict values 
	W = tf.get_variable("W",  [1], dtype=tf.float64) 
	b = tf.get_variable("b",  [1], dtype=tf.float64) 
	y = W*features['x']  + b 
	# Loss sub-graph 
	loss = tf.reduce_sum(tf.square(y - labels))  
	# Training sub-graph 
	global_step = tf.train.get_global_step() 
	optimizer = tf.train.GradientDescentOptimizer(0.01) 
	train = tf.group(optimizer.minimize(loss), 
					tf.assign_add(global_step,  1))  
	# EstimatorSpec connects subgraphs we built to the  
	# appropriate functionality.  
	return tf.estimator.EstimatorSpec( 
			mode=mode, 
			predictions=y, 
			loss=loss, 
			train_op=train) 
estimator = tf.estimator.Estimator(model_fn=model_fn)  
# define our data sets 
x_train = np.array([1.,  2.,  3.,  4.]) 
y_train = np.array([0.,  -1.,  -2.,  -3.]) 
x_eval = np.array([2.,  5.,  8.,  1.]) 
y_eval = np.array([-1.01,  -4.1,  -7.,  0.]) 
input_fn = tf.estimator.inputs.numpy_input_fn(  
	{"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True) 
train_input_fn = tf.estimator.inputs.numpy_input_fn(  
	{"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False) 
eval_input_fn = tf.estimator.inputs.numpy_input_fn(  
	{"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)  
	
# train 
estimator.train(input_fn=input_fn, steps=1000)  
# Here we evaluate how well our model did. 
train_metrics = estimator.evaluate(input_fn=train_input_fn) 
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)  
print("train metrics: %r"% train_metrics)  
print("eval metrics: %r"% eval_metrics)
```

 运行结果如下：

``` python
train metrics:  {'loss':  1.227995e-11,  'global_step':  1000}  
eval metrics:  {'loss':  0.01010036,  'global_step':  1000}
```

注意自定义 `model_fn()` 函数的内容与我们从低级别 API 得到的手动训练循环是非常相似的。

## Next steps

现在你已经掌握了 TensorFlow 的基础知识。我们还有更多的教程助你了解学习更多。如果你是一个机器学习初学者，请查看 [MNIST for beginners](https://tensorflow.google.cn/get_started/mnist/beginners) ，否则请查看 [Deep MNIST for experts](https://tensorflow.google.cn/get_started/mnist/pros) 。