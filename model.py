# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import math
from os.path import join
import shutil

# '''
#
#     1. 定义时间步, 定义每个输入数据与前多少个有序的输入的数据有关联
#     2. 定义隐层神经元的数量
#     3. 定义每批训练样本数
#     4. 定义输入层维度、权重、偏置
#     5. 定义输出层维度、权重、偏置
#     6. 定义学习率，学习率越小，越容易陷入局部最优结果，学习率越大，相邻两次训练结果间的抖动越大
#     7. 定义损失函数
#     8. 定义训练次数
#     9. 构建训练数据集
#     10.构建测试数据集
#
# '''

flags = tf.app.flags

flags.DEFINE_integer('train_batch_size', 500, 'train_batch_size')
flags.DEFINE_integer('dev_batch_size', 50, 'dev batch size')
flags.DEFINE_integer('test_batch_size', 500, 'test batch size')
flags.DEFINE_integer('dict_size', 6000, 'dict_size')
flags.DEFINE_integer('category_num', 5, 'category number')
flags.DEFINE_float('learning_rate', 0.005, 'learning_rate')
flags.DEFINE_integer('num_units', 64, 'the number of units in LSTM cell')
flags.DEFINE_integer('num_layer', 2, 'num_layers')
flags.DEFINE_integer('time_step', 32, 'timestep_size')
flags.DEFINE_integer('epoch_num', 50, 'epoch_num')
flags.DEFINE_integer('epochs_per_dev', 2, 'epoch per dev')
flags.DEFINE_integer('epochs_per_test', 100, 'epoch per test')
flags.DEFINE_integer('epochs_per_save', 2, 'epoch per save')
flags.DEFINE_integer('steps_per_print', 100, 'steps per print')
flags.DEFINE_integer('steps_per_summary', 100, 'steps per summary')
flags.DEFINE_integer('embedding_size', 64, 'embedding size')
flags.DEFINE_string('summaries_dir', 'summaries\\', 'summaries dir')
flags.DEFINE_string('checkpoint_dir', 'ckpt\\model.ckpt', 'checkpoint dir')
flags.DEFINE_float('keep_prob', 0.5, 'keep prob dropout')
flags.DEFINE_boolean('train', True, 'train or not')

FLAGS = tf.app.flags.FLAGS


def get_data(data_x, data_y):
    """
    载入数据集并分割
    :param data_x:
    :param data_y:
    :return: Arrays

    """
    print('Data X Length', len(data_x), 'Data Y Length', len(data_y))
    print('Data X Example', data_x[0])
    print('Data Y Example', data_y[0])

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=40)
    train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.2, random_state=40)

    print('Train X Shape', train_x.shape, 'Train Y Shape', train_y.shape)
    print('Dev X Shape', dev_x.shape, 'Dev Y Shape', dev_y.shape)
    print('Test Y Shape', test_x.shape, 'Test Y Shape', test_y.shape)
    return train_x, train_y, dev_x, dev_y, test_x, test_y


def weight(shape, stddev=0.1, mean=0):
    """初始化权重"""
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def bias(shape, value=0.1):
    """初始化偏置"""
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


def lstm_cell(num_units, keep_prob=0.5):
    """定义LSTM核"""
    cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)


def Input_(iterator):
    """
    输入层
    """
    with tf.variable_scope('inputs'):
        x, y_label = iterator.get_next()
    return x, y_label


def Embedding_(X):
    """字嵌入层"""
    with tf.variable_scope('embedding'):
        embedding_matrix = tf.Variable(tf.random_normal([FLAGS.dict_size, FLAGS.embedding_size], -1.0, 1.0))
        # embedding_matrix = tf.Variable(tf.random_normal([5155,200],-1.0,1.0))
        inputs = tf.nn.embedding_lookup(embedding_matrix, X)
    return inputs


def LSTM_(inputs, keep_prob):
    """LSTM层"""
    cell_fw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
    cell_bw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
    """
    本来input是[[a1s1,a1s2,a1s3,...,a1sn],[a2s1,a2s2,...,a2sn],...]
    现在变成[[a1s1,a2s1,a3s1,...,ans1],...,[a1sn,a2sn,a3sn,...,ansn]]
    """
    # shape(time_step, batch_size, input_size )
    inputs = tf.unstack(inputs, FLAGS.time_step, axis=1)  # inputs变形


    output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, dtype=tf.float32)
    output = tf.stack(output, axis=1)  # output变形回来 shape=(batch_size, time_step, 2,  output_size)
    print('Output', output)
    output = tf.reshape(output, [-1, FLAGS.num_units * 2])  # shape = (batch内总字数，2倍units数)
    print('Output Reshape', output)

    return output


def Output_(output):
    """输出层（全连接层），对每个字的输出是一个维度为类别数的向量"""
    with tf.variable_scope('outputs'):
        w = weight([FLAGS.num_units * 2, FLAGS.category_num])
        b = bias([FLAGS.category_num])
        y = tf.matmul(output, w) + b  # shape = (batch内总字数，类别数)
        y_predict = tf.cast(tf.argmax(y, axis=1), tf.int32)
        # print('y_predict', y_predict)
        # tf.cast()用于转换tensor的数据类型
        # tf.argmax()用于提取行或列最大值的位置
    return y, y_predict  # shape = (batch内总字数,)


def prediction(y_predict, y_label_reshape):
    print('y_predict', y_predict, 'y_label_reshape', y_label_reshape)
    # tf.cast(y_label_reshape, tf.float32)
    correct_prediction = tf.equal(y_predict, y_label_reshape)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求平均值
    tf.summary.scalar('accuracy', accuracy)
    print('Prediction', correct_prediction, 'Accuracy', accuracy)
    return accuracy


def count_cross_entropy(y, y_label_reshape):
    # 计算损失函数：交叉熵
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_label_reshape,
        logits=tf.cast(y, tf.float32)))
    tf.summary.scalar('loss', cross_entropy)
    return cross_entropy


def train_(cross_entropy, global_step):
    # learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10, decay_rate=0.98, staircase=True)

    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy, global_step=global_step)


def main():

    data_x = pd.read_csv("train_X.csv")
    data_y = pd.read_csv("train_Y.csv")

    data_x = data_x.astype(int)
    data_y = data_y.astype(int)
    data_x = data_x.values
    data_y = data_y.values

    train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(data_x, data_y)
    print('Vocab Size', FLAGS.dict_size)

    global_step = tf.Variable(-1, trainable=False, name='global_step')  # 这是啥

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.batch(FLAGS.train_batch_size)

    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_x, dev_y))
    dev_dataset = dev_dataset.batch(FLAGS.dev_batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_dataset = test_dataset.batch(FLAGS.test_batch_size)

    train_steps = math.ceil(train_x.shape[0] / FLAGS.train_batch_size)
    dev_steps = math.ceil(dev_x.shape[0] / FLAGS.dev_batch_size)
    test_steps = math.ceil(test_x.shape[0] / FLAGS.test_batch_size)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    train_initializer = iterator.make_initializer(train_dataset)
    dev_initializer = iterator.make_initializer(dev_dataset)
    test_initializer = iterator.make_initializer(test_dataset)
    x, y_label = Input_(iterator)
    inputs = Embedding_(x)
    keep_prob = tf.placeholder(tf.float32, [])
    output = LSTM_(inputs, keep_prob)
    y, y_predict = Output_(output)
    y_label_reshape = tf.cast(tf.reshape(y_label, [-1]), tf.int32)
    cross_entropy = count_cross_entropy(y, y_label_reshape)
    accuracy = prediction(y_predict, y_label_reshape)
    train = train_(cross_entropy, global_step)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        gstep = 0

        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(join(FLAGS.summaries_dir, 'train'),
                                       sess.graph)

        if FLAGS.train:

            if tf.gfile.Exists(FLAGS.summaries_dir):
                shutil.rmtree(FLAGS.summaries_dir, ignore_errors=True)

            for epoch in range(FLAGS.epoch_num):
                tf.train.global_step(sess, global_step_tensor=global_step)

                sess.run(train_initializer)

                for step in range(int(train_steps)):
                    smrs, loss, acc, gstep, _ = sess.run([summaries, cross_entropy, accuracy, global_step, train],
                                                         feed_dict={keep_prob: FLAGS.keep_prob})

                    if step % FLAGS.steps_per_print == 0:
                        print('Global Step', gstep, 'Step', step, 'Train Loss', loss, 'Accuracy', acc)

                    # if gstep % FLAGS.steps_per_summary == 0:
                    #     writer.add_summary(smrs, gstep)
                    #     print('Write summaries to', FLAGS.summaries_dir)
                    #
                if epoch % FLAGS.epochs_per_dev == 0:
                    # Dev
                    sess.run(dev_initializer)
                    for step in range(int(dev_steps)):
                        if step % FLAGS.steps_per_print == 0:
                            print('Dev Accuracy', sess.run(accuracy, feed_dict={keep_prob: 1}), 'Step', step)

                if epoch % FLAGS.epochs_per_save == 0:
                    # 保存模型
                    saver.save(sess, FLAGS.checkpoint_dir, global_step=gstep)


if __name__ == '__main__':

    main()
