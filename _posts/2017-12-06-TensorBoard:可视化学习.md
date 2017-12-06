---
layout:     post
title:      TensorBoard：可视化学习
subtitle:   千里之行始于足下
date:       2017-12-06
author:     YQY
header-img: img/post_tensorflow_7.jpg
catalog: true
tags:
    - Tensorflow

---

> 我是一个很懒的人，我想试试
>
> 希望我能坚持到最后，把tensorflow的官方教程全部翻译出来
>
> 提高自己，也帮助他人

# TensorBoard: Visualizing Learning

使用 TensorFlow 计算，如训练一个大型的深度神经网络，可能会非常的复杂与难以理解。为了使其容易理解，调试和优化 TensorFlow 程序，我们有一套叫做 TensorBoard 的可视化工具。你可以使用 TensorBoard 可视化你的 TensorFlow 图，绘制图执行的量化指标，并通过它以图像的形式显示附加的数据。当 TensorBoard 设置完成后，它看起来就像下面这样：

![MNIST TensorBoard](https://www.tensorflow.org/images/mnist_tensorboard.png)

本教程旨在教会你入门简单的 TensorBoard 用法。还有其他的资源是可用的。如  [TensorBoard's GitHub](https://github.com/tensorflow/tensorboard)  有很多关于 TensorBoard 使用的信息，包括了提示，技巧以及调试信息。

## Serializing the data

TensorBoard 通过读取 TensorFlow 事件文件来操作，它包括了运行 TensorFlow 过程中生成的汇总数据。以下是 TensorBoard 在一般的生命周期中产生的汇总数据。

首先，创建你想要收集汇总数据的 TensorFlow 图，并决定你想要哪个节点使用 [汇总操作](https://www.tensorflow.org/api_guides/python/summary) 注释(annotate)。

例如，假设你要训练一个卷积神经网络识别 MNIST 数字，你想要记录在整个训练过程中学习率是如何变化的，目标函数是如何变化的。通过向节点附加 [`tf.summary.scalar`](https://www.tensorflow.org/api_docs/python/tf/summary/scalar) 操作来收集这些数据，并分别输出学习率和损失。然后给每个 `scalar_summary`  一个有意义的 `tag`，如`'learning rate'` or `'loss function'` 。

也许你想要可视化一个特定层的激活分布，或者梯度或者权重的分布。可以通过分别向梯度输出和权重变量附加 [`tf.summary.histogram`](https://www.tensorflow.org/api_docs/python/tf/summary/histogram) 操作来收集这些数据。

有关可用的汇总操作的所有详细信息，请查阅有关文档[summary operations](https://www.tensorflow.org/api_guides/python/summary) 。

在 TensorFlow 中，操作只有当你执行或者一个操作依赖它的输出时才会执行。我们创建的汇总节点(summary nodes)都围绕在你的图的边上：没有一个操作的运行是依赖它们的。所以，为了生成汇总节点，我们需要运行所有的汇总节点。手动管理这些是很乏味的，所以使用 [`tf.summary.merge_all`](https://www.tensorflow.org/api_docs/python/tf/summary/merge_all)  联合它们到一个操作来生成所有的汇总数据。

然后你可以只执行合并汇总操作，它将在一个给定的步骤将所有的汇总数据生成一个序列化的`Summary`  protobuf 对象。最后，为了将这个汇总数据写入到硬盘，传递汇总 protobuf  到[`tf.summary.FileWriter`](https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter) 。

`FileWriter` 的构造函数是带有 logdir 参数的——这个 logdir 非常重要，是所有的事件输出的目录。`FileWriter` 在构造函数中也可以使用`Graph` 这个可选参数。如果接收到一个`Graph` 对象，那么 TensorBoard 将随着张量 shape 信息可视化 graph。这将使你更好的理解图中运行情况：请查阅 [Tensor shape information](https://www.tensorflow.org/get_started/graph_viz#tensor_shape_information) 。

注意你已经修改了你的 graph，并有了一个 `FileWriter` ，你已准备开始运行你的网络了！如果你需要，你可以在每一步中运行合并汇总操作并保存大量的训练数据。这样你将得到超出你需要的很多的数据，相反你可以考虑每 `n` 个步骤运行一次合并汇总操作。

以下代码实例修改于 [simple MNIST tutorial](https://www.tensorflow.org/get_started/mnist/beginners) ，我们只是增加了一些汇总操作，并每十个步骤运行一次它们。如果你运行这个并启动 `tensorboard --logdir=/tmp/tensorflow/mnist` ，你将能够可视化这些统计数据，例如在训练期间，权重或者准确率是如何变化的。以下只是代码的部分，完整代码 [在这](https://www.github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py) 。

```python
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', keep_prob)
  dropped = tf.nn.dropout(hidden1, keep_prob)

# Do not apply softmax activation yet, see below.
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
  # The raw formulation of cross-entropy,
  #
  # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                               reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the
  # raw outputs of the nn_layer above, and then average across
  # the batch.
  diff = tf.nn.softmax_cross_entropy_with_logits(targets=y_, logits=y)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
      cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
tf.global_variables_initializer().run()
```

在我们初始化 `FileWriters` 之后，我们必须在我们训练和测试模型时附加汇总到 `FileWriters` 。

```python
# Train the model, and also write summaries.
# Every 10th step, measure test-set accuracy, and write test summaries
# All other steps, run train_step on training data, & add training summaries

def feed_dict(train):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train or FLAGS.fake_data:
    xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
    k = FLAGS.dropout
  else:
    xs, ys = mnist.test.images, mnist.test.labels
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}

for i in range(FLAGS.max_steps):
  if i % 10 == 0:  # Record summaries and test-set accuracy
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  else:  # Record train set summaries, and train
    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_writer.add_summary(summary, i)
```

你现在已经准备好使用 TensorBoard 可视化这些数据。

## Launching TensorBoard

运行 TensorBoard，请使用以下命令(或者 `python -m tensorboard.main`) 

```python
tensorboard --logdir=path/to/log-directory
```

其中`logdir` 指向`FileWriter` 序列化其数据的目录。如果`logdir` 目录包含单独运行的序列化数据的子目录，TensorBoard 将可视化所有的这些运行的数据。一旦 TensorBoard 运行起来，输入 `localhost:6006` 到你的 web 浏览器，已查看 TensorBoard。

当浏览 TensorBoard 时，你将在右上角看到导航标签。每个标签表示可以可视化的一系列序列化数据。

有关如何使用 *graph* 选项卡可视化图的更多信息，请查阅 [TensorBoard: Graph Visualization](https://www.tensorflow.org/get_started/graph_viz) 。

有关 TensorBoard 的更多的使用信息，请查阅 [TensorBoard's GitHub](https://github.com/tensorflow/tensorboard) 。
