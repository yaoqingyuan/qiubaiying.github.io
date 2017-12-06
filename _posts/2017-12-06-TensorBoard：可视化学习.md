---
layout:     post
title:      TensorBoard�����ӻ�ѧϰ
subtitle:   ǧ��֮��ʼ������
date:       2017-12-06
author:     YQY
header-img: img/post_tensorflow_7.jpg
catalog: true
tags:
    - Tensorflow

---

> ����һ���������ˣ���������
>
> ϣ�����ܼ�ֵ���󣬰�tensorflow�Ĺٷ��̳�ȫ���������
>
> ����Լ���Ҳ��������

# TensorBoard: Visualizing Learning

ʹ�� TensorFlow ���㣬��ѵ��һ�����͵���������磬���ܻ�ǳ��ĸ�����������⡣Ϊ��ʹ��������⣬���Ժ��Ż� TensorFlow ����������һ�׽��� TensorBoard �Ŀ��ӻ����ߡ������ʹ�� TensorBoard ���ӻ���� TensorFlow ͼ������ͼִ�е�����ָ�꣬��ͨ������ͼ�����ʽ��ʾ���ӵ����ݡ��� TensorBoard ������ɺ�����������������������

![MNIST TensorBoard](https://www.tensorflow.org/images/mnist_tensorboard.png)

���̳�ּ�ڽ̻������ż򵥵� TensorBoard �÷���������������Դ�ǿ��õġ���  [TensorBoard's GitHub](https://github.com/tensorflow/tensorboard)  �кܶ���� TensorBoard ʹ�õ���Ϣ����������ʾ�������Լ�������Ϣ��

## Serializing the data

TensorBoard ͨ����ȡ TensorFlow �¼��ļ��������������������� TensorFlow ���������ɵĻ������ݡ������� TensorBoard ��һ������������в����Ļ������ݡ�

���ȣ���������Ҫ�ռ��������ݵ� TensorFlow ͼ������������Ҫ�ĸ��ڵ�ʹ�� [���ܲ���](https://www.tensorflow.org/api_guides/python/summary) ע��(annotate)��

���磬������Ҫѵ��һ�����������ʶ�� MNIST ���֣�����Ҫ��¼������ѵ��������ѧϰ������α仯�ģ�Ŀ�꺯������α仯�ġ�ͨ����ڵ㸽�� [`tf.summary.scalar`](https://www.tensorflow.org/api_docs/python/tf/summary/scalar) �������ռ���Щ���ݣ����ֱ����ѧϰ�ʺ���ʧ��Ȼ���ÿ�� `scalar_summary`  һ��������� `tag`����`'learning rate'` or `'loss function'` ��

Ҳ������Ҫ���ӻ�һ���ض���ļ���ֲ��������ݶȻ���Ȩ�صķֲ�������ͨ���ֱ����ݶ������Ȩ�ر������� [`tf.summary.histogram`](https://www.tensorflow.org/api_docs/python/tf/summary/histogram) �������ռ���Щ���ݡ�

�йؿ��õĻ��ܲ�����������ϸ��Ϣ��������й��ĵ�[summary operations](https://www.tensorflow.org/api_guides/python/summary) ��

�� TensorFlow �У�����ֻ�е���ִ�л���һ�����������������ʱ�Ż�ִ�С����Ǵ����Ļ��ܽڵ�(summary nodes)��Χ�������ͼ�ı��ϣ�û��һ���������������������ǵġ����ԣ�Ϊ�����ɻ��ܽڵ㣬������Ҫ�������еĻ��ܽڵ㡣�ֶ�������Щ�Ǻܷ�ζ�ģ�����ʹ�� [`tf.summary.merge_all`](https://www.tensorflow.org/api_docs/python/tf/summary/merge_all)  �������ǵ�һ���������������еĻ������ݡ�

Ȼ�������ִֻ�кϲ����ܲ�����������һ�������Ĳ��轫���еĻ�����������һ�����л���`Summary`  protobuf �������Ϊ�˽������������д�뵽Ӳ�̣����ݻ��� protobuf  ��[`tf.summary.FileWriter`](https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter) ��

`FileWriter` �Ĺ��캯���Ǵ��� logdir �����ġ������ logdir �ǳ���Ҫ�������е��¼������Ŀ¼��`FileWriter` �ڹ��캯����Ҳ����ʹ��`Graph` �����ѡ������������յ�һ��`Graph` ������ô TensorBoard ���������� shape ��Ϣ���ӻ� graph���⽫ʹ����õ����ͼ���������������� [Tensor shape information](https://www.tensorflow.org/get_started/graph_viz#tensor_shape_information) ��

ע�����Ѿ��޸������ graph��������һ�� `FileWriter` ������׼����ʼ������������ˣ��������Ҫ���������ÿһ�������кϲ����ܲ��������������ѵ�����ݡ������㽫�õ���������Ҫ�ĺܶ�����ݣ��෴����Կ���ÿ `n` ����������һ�κϲ����ܲ�����

���´���ʵ���޸��� [simple MNIST tutorial](https://www.tensorflow.org/get_started/mnist/beginners) ������ֻ��������һЩ���ܲ�������ÿʮ����������һ�����ǡ������������������� `tensorboard --logdir=/tmp/tensorflow/mnist` ���㽫�ܹ����ӻ���Щͳ�����ݣ�������ѵ���ڼ䣬Ȩ�ػ���׼ȷ������α仯�ġ�����ֻ�Ǵ���Ĳ��֣��������� [����](https://www.github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py) ��

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

�����ǳ�ʼ�� `FileWriters` ֮�����Ǳ���������ѵ���Ͳ���ģ��ʱ���ӻ��ܵ� `FileWriters` ��

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

�������Ѿ�׼����ʹ�� TensorBoard ���ӻ���Щ���ݡ�

## Launching TensorBoard

���� TensorBoard����ʹ����������(���� `python -m tensorboard.main`) 

```python
tensorboard --logdir=path/to/log-directory
```

����`logdir` ָ��`FileWriter` ���л������ݵ�Ŀ¼�����`logdir` Ŀ¼�����������е����л����ݵ���Ŀ¼��TensorBoard �����ӻ����е���Щ���е����ݡ�һ�� TensorBoard �������������� `localhost:6006` ����� web ��������Ѳ鿴 TensorBoard��

����� TensorBoard ʱ���㽫�����Ͻǿ���������ǩ��ÿ����ǩ��ʾ���Կ��ӻ���һϵ�����л����ݡ�

�й����ʹ�� *graph* ѡ����ӻ�ͼ�ĸ�����Ϣ������� [TensorBoard: Graph Visualization](https://www.tensorflow.org/get_started/graph_viz) ��

�й� TensorBoard �ĸ����ʹ����Ϣ������� [TensorBoard's GitHub](https://github.com/tensorflow/tensorboard) ��
