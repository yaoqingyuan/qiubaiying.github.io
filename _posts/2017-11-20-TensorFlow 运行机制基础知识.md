---
layout:     post
title:      TensorFlow 运行机制基础知识
subtitle:   千里之行始于足下
date:       2017-11-20
author:     YQY
header-img: img/post_tensorflow_4.jpg
catalog: true
tags:
    - Tensorflow
---

> 我是一个很懒的人，我想试试
>
> 希望我能坚持到最后，把tensorflow的官方教程全部翻译出来
>
> 提高自己，也帮助他人

# TensorFlow Mechanics 101

代码：[tensorflow/examples/tutorials/mnist/](https://www.github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/)

本教程的目标是向大家展示在(经典)的 MNIST 数据集下如何使用 TensorFlow 训练和评估一个用于识别手写数字的简单的前馈神经网络(feed-forward neural network)。本教程的目标读者是有兴趣使用 TensorFlow 的有经验的机器学习使用者。

这些教程不适用于机器学习初学者教学。

请确保你已经按照说明[安装 TensorFlow](https://tensorflow.google.cn/install/index)。

## Tutorial Files

本教程引用了以下文件：

| 文件                                       | 目的                                       |
| ---------------------------------------- | ---------------------------------------- |
| [`mnist.py`](https://www.github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist.py) | 构建一个全连接 MNIST 模型代码                       |
| [`fully_connected_feed.py`](https://www.github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/fully_connected_feed.py) | 主要代码是利用 feed dictionary 对构建的 MNIST 模型使用下载数据集进行训练 |

直接运行 `fully_connected_feed.py` 文件开始训练：

```python
python fully_connected_feed.py
```

## Prepare the Data

在机器学习中 MNIST 是一个经典问题——通过观察一个28x28像素的灰度手写数字图像来确定图上的数字表示几，其中所有的数字都是从0到9.

![MNIST Digits](https://tensorflow.google.cn/images/mnist_digits.png)

更多信息，请参考 [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/) 或 [Chris Olah's visualizations of MNIST](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)。

### Download

在 `run_training()`方法之前，`input_data.read_data_sets()`函数确保了正确的数据集被下载到你的本地的训练目录，然后解压并返回`DataSet`实例的字典。

```python
data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
```

**注意**: `fake_data` 标志是用于单元测试的，读者可以忽略。

| 数据集                    | 目的                      |
| ---------------------- | ----------------------- |
| `data_sets.train`      | 55000 个图像和标志，用于训练       |
| `data_sets.validation` | 5000 个图像和标志，迭代验证训练准确度   |
| `data_sets.test`       | 10000 图像和标志，用于最终测试训练准确度 |

### Inputs and Placeholders

 `placeholder_inputs()` 函数创建两个 [`tf.placeholder`](https://tensorflow.google.cn/api_docs/python/tf/placeholder) 操作，定义了输入的形状，包括 `batch_size`，到图的其余部分，并将实际训练样本输入到其中。

```python
images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                       mnist.IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
```

再往下，在训练循环的每一步中，完整的图像和标签数据集被分割成符合`batch_size`的大小，并与占位符操作相匹配。然后使用`feed_dict`参数将数据传入`sess.run()`函数。

## Build the Graph

为数据创建占位符后，从`mnist.py` 文件中根据三个阶段模型：`inference()`，`loss()`，和 `training()`就可以构建图了。

1. `inference()` - 根据需求构建图，以便向前运行网络进行预测。
2. `loss()` - 在 inference 图中添加产生损失(loss)所需的操作(ops)。
3. `training()` - 在 loss 图中添加计算和应用梯度(gradients)的操作。

![img](https://tensorflow.google.cn/images/mnist_subgraph.png)

### Inference

根据需要 `inference()` 函数构建好图并返回包含预测结果(output predictions)的张量(tensor)。

它将图像占位符作为输入，并在其上构建一对全连接层，其中有一个 [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) 激活函数，其后接着是十个指定输出logits的线性层节点。

每一层都在唯一的 [`tf.name_scope`](https://tensorflow.google.cn/api_docs/python/tf/name_scope)之下创建，它充当在该范围内创建的项目的前缀。

```python
with tf.name_scope('hidden1'):
```

在定义的范围内，每个层所使用的权重和偏置都在 [`tf.Variable`](https://tensorflow.google.cn/api_docs/python/tf/Variable)的实例中生成，并具有所需的形状：

```python
weights = tf.Variable(
    tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                        stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
    name='weights')
biases = tf.Variable(tf.zeros([hidden1_units]),
                     name='biases')
```

例如，当它们在`hidden1`范围内创建，赋予权重变量唯一的名称就是 "`hidden1/weights`"。

每个变量作为他们构造的一部分都会被赋予初始化操作。

在这种最常见的情况下，通过[`tf.truncated_normal`](https://tensorflow.google.cn/api_docs/python/tf/truncated_normal) 来初始化权重，并给予他们2维形状的张量，其中第一维表示该层中权重连接的单元数量，而其中第二维表示该层权重连接到的单元数量。对于名为 `hidden1`的第一层，维度为 `[IMAGE_PIXELS, hidden1_units]`，因为权重连接输入图像到 hidden1 层。根据给定的均值和标准差 `tf.truncated_normal` 初始化生成一个随机分布。

然后通过 [`tf.zeros`](https://tensorflow.google.cn/api_docs/python/tf/zeros) 初始化偏置，以确保他们使用全为零的值开始，而它们的形状就是该层连接的所有单元数量。

图的三个主要操作——两个 [`tf.nn.relu`](https://tensorflow.google.cn/api_docs/python/tf/nn/relu)操作，在隐藏层中包含了 [`tf.matmul`](https://tensorflow.google.cn/api_docs/python/tf/matmul) 。一个 logits 所需的额外的`tf.matmul`操作。然后依次创建，而每个单独的 `tf.Variable`实例连接到每个输入占位符或前一层的输出张量。

```python
hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
```

```python
hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
```

```python
logits = tf.matmul(hidden2, weights) + biases
```

最后，返回包含输出的 `logits` 张量。

### Loss

`loss()` 函数通过添加所需的损失操作来进一步构建图。

首先，来自`labels_placeholder`的值被转化为64位整数。然后增加一个 [`tf.nn.sparse_softmax_cross_entropy_with_logits`](https://tensorflow.google.cn/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits) 操作到`labels_placeholder`自动产生的1-hot 标签，并比较从 `inference()`函数中的输出 logits 与这些1-hot 标签。

```python
labels = tf.to_int64(labels)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels, logits=logits, name='xentropy')
```

然后使用[`tf.reduce_mean`](https://tensorflow.google.cn/api_docs/python/tf/reduce_mean)计算 batch 维度(第一维)的平均交叉熵的值，并作为总损失。

```python
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
```

然后返回包含损失值的张量。

> **注意：** 交叉熵是信息论的一个概念，它允许我们基于实际是真的，描述相信神经网络是有多糟。欲了解更多信息，请阅读视觉信息理论博客 (http://colah.github.io/posts/2015-09-Visual-Information/)

### Training

`training()` 函数添加通过 [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) 最小化损失的操作。

首先，从`loss()`函数中获取损失张量，并传递给 [`tf.summary.scalar`](https://tensorflow.google.cn/api_docs/python/tf/summary/scalar)，然后再使用 [`tf.summary.FileWriter`](https://tensorflow.google.cn/api_docs/python/tf/summary/FileWriter) (见下文)，向事件文件(events file)生成汇总值(summary values)。在这里，每次写入汇总值时，它都会发出当前的损失值(snapshot value)。

```python
tf.summary.scalar('loss', loss)
```

接下来，我们实例化一个 [`tf.train.GradientDescentOptimizer`](https://tensorflow.google.cn/api_docs/python/tf/train/GradientDescentOptimizer) ，负责按照我们要求的学习率(learning rate)，应用梯度下降(gradients)。

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
```

再然后，我们生成一个变量来保存全局训练步骤的数值(global training step)，并使用[`tf.train.Optimizer.minimize`](https://tensorflow.google.cn/api_docs/python/tf/train/Optimizer#minimize) 操作来更新系统的训练权重和增加全局训练步骤。按照惯例，这个操作被称为 `train_op`，是 TensorFlow  会话执行一整个训练步骤(见下文)必须执行的操作。

```python
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
```

## Train the Model

一旦图构建完成，就可以通过 `fully_connected_feed.py`文件中用户代码进行循环迭代训练和评估控制。

### The Graph

在 `run_training()` 函数之前是一个 python 的 `with` 命令，它表明所有的构建操作都与默认的全局[`tf.Graph`](https://tensorflow.google.cn/api_docs/python/tf/Graph)实例关联起来。

```python
with tf.Graph().as_default():
```

`tf.Graph`是一个操作集合，将作为一个组(group)一起被执行。大多数的 TensorFlow 使用都只需要依赖于一个默认的图的实例即可。

使用多个图实现更复杂的情况是可能的，但是这超出了本篇教程的范围。

### The Session

一旦所有的构建准备都完成，并且生成了所有的必须的操作，就可以创建 [`tf.Session`](https://tensorflow.google.cn/api_docs/python/tf/Session) 来执行图。

```python
sess = tf.Session()
```

另外，也可以在 `with` 块中生成 `Session` 来限制作用域：

```python
with tf.Session() as sess:
```

无参 session 函数表明这段代码将连接(或者如果没有创建那就创建)默认的本地 session。

在创建 session 后立即在它们的初始化操作中调用[`tf.Session.run`](https://tensorflow.google.cn/api_docs/python/tf/Session#run) 来初始化所有的`tf.Variable` 实例。

```python
init = tf.global_variables_initializer()
sess.run(init)
```

 [`tf.Session.run`](https://tensorflow.google.cn/api_docs/python/tf/Session#run) 方法将运行与作为参数传递的操作(op)对应的完整子图。在首次调用时， `init`操作是一个只包含变量初始化的[`tf.group`](https://tensorflow.google.cn/api_docs/python/tf/group) 。图的其他部分不会在这里执行，而只会在下面的循环训练里执行。

### Train Loop

在 session 中初始化了所有的变量后，训练就可以开始了。

训练中的每一步都是由用户代码控制着，而有效训练的最简单循环如下：

```python
for step in xrange(FLAGS.max_steps):
    sess.run(train_op)
```

然而，本教程会稍微复杂一点点，因为它必须在每个循环中将输入数据切片以匹配先前生成的占位符。

#### Feed the Graph

对于每一步，代码将生成一个 反馈字典(feed dictionary)，其中包含该步骤训练的样本，例子的键(key)就是它们代表的占位符操作。

在`fill_feed_dict()`函数中，查询给定的`DataSet`的下一批`batch_size`的图像和标签，与占位符相匹配的张量包含了下一批的图像和标签。

```python
images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                               FLAGS.fake_data)
```

A python dictionary object is then generated with the placeholders as keys and the representative feed tensors as values.

```python
feed_dict = {
    images_placeholder: images_feed,
    labels_placeholder: labels_feed,
}
```

This is passed into the `sess.run()` function's `feed_dict` parameter to provide the input examples for this step of training.

#### Check the Status

The code specifies two values to fetch in its run call: `[train_op, loss]`.

```python
for step in xrange(FLAGS.max_steps):
    feed_dict = fill_feed_dict(data_sets.train,
                               images_placeholder,
                               labels_placeholder)
    _, loss_value = sess.run([train_op, loss],
                             feed_dict=feed_dict)
```

Because there are two values to fetch, `sess.run()` returns a tuple with two items. Each `Tensor` in the list of values to fetch corresponds to a numpy array in the returned tuple, filled with the value of that tensor during this step of training. Since `train_op` is an `Operation` with no output value, the corresponding element in the returned tuple is `None` and, thus, discarded. However, the value of the `loss` tensor may become NaN if the model diverges during training, so we capture this value for logging.

Assuming that the training runs fine without NaNs, the training loop also prints a simple status text every 100 steps to let the user know the state of training.

```python
if step % 100 == 0:
    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
```

#### Visualize the Status

In order to emit the events files used by [TensorBoard](https://tensorflow.google.cn/get_started/summaries_and_tensorboard), all of the summaries (in this case, only one) are collected into a single Tensor during the graph building phase.

```python
summary = tf.summary.merge_all()
```

And then after the session is created, a [`tf.summary.FileWriter`](https://tensorflow.google.cn/api_docs/python/tf/summary/FileWriter) may be instantiated to write the events files, which contain both the graph itself and the values of the summaries.

```python
summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
```

Lastly, the events file will be updated with new summary values every time the `summary` is evaluated and the output passed to the writer's `add_summary()` function.

```python
summary_str = sess.run(summary, feed_dict=feed_dict)
summary_writer.add_summary(summary_str, step)
```

When the events files are written, TensorBoard may be run against the training folder to display the values from the summaries.

![MNIST TensorBoard](https://tensorflow.google.cn/images/mnist_tensorboard.png)

**NOTE**: For more info about how to build and run Tensorboard, please see the accompanying tutorial [Tensorboard: Visualizing Learning](https://tensorflow.google.cn/get_started/summaries_and_tensorboard).

#### Save a Checkpoint

In order to emit a checkpoint file that may be used to later restore a model for further training or evaluation, we instantiate a [`tf.train.Saver`](https://tensorflow.google.cn/api_docs/python/tf/train/Saver).

```python
saver = tf.train.Saver()
```

In the training loop, the [`tf.train.Saver.save`](https://tensorflow.google.cn/api_docs/python/tf/train/Saver#save) method will periodically be called to write a checkpoint file to the training directory with the current values of all the trainable variables.

```python
saver.save(sess, FLAGS.train_dir, global_step=step)
```

At some later point in the future, training might be resumed by using the [`tf.train.Saver.restore`](https://tensorflow.google.cn/api_docs/python/tf/train/Saver#restore) method to reload the model parameters.

```python
saver.restore(sess, FLAGS.train_dir)
```

## Evaluate the Model

Every thousand steps, the code will attempt to evaluate the model against both the training and test datasets. The `do_eval()` function is called thrice, for the training, validation, and test datasets.

```python
print('Training Data Eval:')
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.train)
print('Validation Data Eval:')
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.validation)
print('Test Data Eval:')
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.test)
```

> Note that more complicated usage would usually sequester the `data_sets.test` to only be checked after significant amounts of hyperparameter tuning. For the sake of a simple little MNIST problem, however, we evaluate against all of the data.

### Build the Eval Graph

Before entering the training loop, the Eval op should have been built by calling the `evaluation()` function from `mnist.py` with the same logits/labels parameters as the `loss()` function.

```python
eval_correct = mnist.evaluation(logits, labels_placeholder)
```

The `evaluation()` function simply generates a [`tf.nn.in_top_k`](https://tensorflow.google.cn/api_docs/python/tf/nn/in_top_k) op that can automatically score each model output as correct if the true label can be found in the K most-likely predictions. In this case, we set the value of K to 1 to only consider a prediction correct if it is for the true label.

```python
eval_correct = tf.nn.in_top_k(logits, labels, 1)
```

### Eval Output

One can then create a loop for filling a `feed_dict` and calling `sess.run()` against the `eval_correct` op to evaluate the model on the given dataset.

```python
for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
```

The `true_count` variable simply accumulates all of the predictions that the `in_top_k` op has determined to be correct. From there, the precision may be calculated from simply dividing by the total number of examples.

```python
precision = true_count / num_examples
print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
      (num_examples, true_count, precision))
```
