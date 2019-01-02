# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.model_selection import train_test_split
import math
from os.path import join
import shutil
import numpy as np
tf.reset_default_graph()


flags = tf.app.flags

flags.DEFINE_integer('train_batch_size', 256, 'train_batch_size')
flags.DEFINE_integer('dev_batch_size', 256, 'dev batch size')
flags.DEFINE_integer('test_batch_size', 1, 'test batch size')
flags.DEFINE_integer('dict_size', 6000, 'dict_size')
flags.DEFINE_integer('category_num', 5, 'category number')
flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
flags.DEFINE_integer('num_units', 64, 'the number of units in LSTM cell')
flags.DEFINE_integer('num_layer', 2, 'num_layers')
flags.DEFINE_integer('time_step', 70, 'timestep_size')
flags.DEFINE_integer('epoch_num', 28, 'epoch_num')
flags.DEFINE_integer('epochs_per_dev', 1, 'epoch per dev')
flags.DEFINE_integer('epochs_per_save', 4, 'epoch per save')
flags.DEFINE_integer('steps_per_print', 200, 'steps per print')
flags.DEFINE_integer('steps_per_summary', 300, 'steps per summary')
flags.DEFINE_integer('embedding_size', 64, 'embedding size')
flags.DEFINE_string('summaries_dir', '..\\summaries\\', 'summaries dir')
flags.DEFINE_string('checkpoint_dir', '..\\ckpt\\model.ckpt', 'checkpoint dir')
flags.DEFINE_float('keep_prob', 0.5, 'keep prob dropout')
flags.DEFINE_boolean('train', False, 'train or test')

FLAGS = tf.app.flags.FLAGS


# 载入数据集并分割
def get_data(data_x, data_y):
    print('Data X Length', len(data_x), 'Data Y Length', len(data_y))
    print('Data X Example', data_x[0])
    print('Data Y Example', data_y[0])

    train_x, dev_x, train_y, dev_y = train_test_split(data_x, data_y, test_size=0.1, random_state=40)

    print('Train X Shape', train_x.shape, 'Train Y Shape', train_y.shape)
    print('Dev X Shape', dev_x.shape, 'Dev Y Shape', dev_y.shape)
    return train_x, train_y, dev_x, dev_y


def padding(data_x, maxlen):
    padded_x = []
    for line in data_x:
        if len(line) <= maxlen:
            padded_x.append(line + [0 for _ in range(maxlen - len(line))])
        else:
            padded_x.append(line[:maxlen])
    return np.array(padded_x, dtype=np.int32)


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


def count_accuracy(y_predict, y_label):
    print('y_predict', y_predict, 'y_label_reshape', y_label)
    # tf.cast(y_label_reshape, tf.float32)
    correct_prediction = tf.equal(y_predict, y_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求平均值
    tf.summary.scalar('accuracy', accuracy)
    print('Prediction', correct_prediction, 'Accuracy', accuracy)
    return accuracy


def main(outputfile):
    # load data
    if FLAGS.train:

        print("Loading training data...")
        data_x = np.load("../data/train_X.npy")
        data_y = np.load("../data/train_Y.npy")

        data_x = padding(data_x, FLAGS.time_step)
        data_y = padding(data_y, FLAGS.time_step)

        train_x, train_y, dev_x, dev_y = get_data(data_x, data_y)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_dataset = train_dataset.shuffle(1000).batch(FLAGS.train_batch_size)

        dev_dataset = tf.data.Dataset.from_tensor_slices((dev_x, dev_y))
        dev_dataset = dev_dataset.shuffle(1000).batch(FLAGS.dev_batch_size)
        train_steps = math.ceil(train_x.shape[0] / FLAGS.train_batch_size)
        dev_steps = math.ceil(dev_x.shape[0] / FLAGS.dev_batch_size)
        print('Building iterator...')
        train_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_initializer = train_iter.make_initializer(train_dataset)
        dev_initializer = train_iter.make_initializer(dev_dataset)
    else:
        print('Loading test data...')
        test_x = np.load("../data/test_X.npy")
        test_x = padding(test_x, FLAGS.time_step)
        print('test_X shape', test_x.shape)
        # test_y = np.zeros(test_x.shape)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_x)
        print('test_X shape', test_dataset.output_shapes)
        test_dataset = test_dataset.batch(FLAGS.test_batch_size)
        test_steps = math.ceil(test_x.shape[0] / FLAGS.test_batch_size)
        print('Building iterator...')
        test_iter = test_dataset.make_initializable_iterator()

        # test_initializer = test_iter.make_initializer(test_dataset)

    global_step = tf.Variable(-1, trainable=False, name='global_step')

    # build model
    print('Building model...')
    # input layer
    with tf.variable_scope('inputs'):
        if FLAGS.train:
            x, y_label = train_iter.get_next()  # get_next()函数取得一批batchsize大小的样本
        else:
            x = test_iter.get_next()

    # embedding layer
    with tf.variable_scope('embedding'):
        embedding_matrix = tf.Variable(tf.random_normal([FLAGS.dict_size, FLAGS.embedding_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(embedding_matrix, x)

    # LSTM layer
    keep_prob = tf.placeholder(tf.float32, [])
    cell_fw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
    cell_bw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
    inputs = tf.unstack(inputs, FLAGS.time_step, axis=1)
    output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, dtype=tf.float32)
    output = tf.stack(output, axis=1)  # output变形回来 shape=(batch_size, time_step, 2,  output_size)
    print('Output', output)
    output = tf.reshape(output, [-1, FLAGS.num_units * 2])  # shape = (batch内总字数，2倍units数)
    print('Output Reshape', output)

    # output layer
    with tf.variable_scope('outputs'):
        w = weight([FLAGS.num_units * 2, FLAGS.category_num])
        b = bias([FLAGS.category_num])
        y = tf.matmul(output, w) + b  # shape = (batch内总字数，类别数)
        y_scores = tf.reshape(y, [-1, FLAGS.time_step, FLAGS.category_num])
        print('y scores', y_scores)

    if FLAGS.train:
        seq_lens = tf.fill([FLAGS.train_batch_size], FLAGS.time_step)
    else:
        seq_lens = tf.fill([FLAGS.test_batch_size], FLAGS.time_step)
    # inference
    if not FLAGS.train:
        with tf.variable_scope('tag_inf'):
            transition_params = tf.get_variable('transitions', shape=[FLAGS.category_num,FLAGS.category_num])
        y_predict,_ = tf.contrib.crf.crf_decode(y_scores, transition_params,seq_lens)
        print('y predict', y_predict)

    # decode and count loss
    else:
        with tf.variable_scope('tag_inf'):
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(y_scores, y_label,seq_lens)
        y_predict, _= tf.contrib.crf.crf_decode(y_scores, transition_params,seq_lens)

        with tf.variable_scope('loss'):
            loss_loglikelihood = tf.reduce_mean(-log_likelihood)
            tf.summary.scalar('loss', loss_loglikelihood)

        accuracy = count_accuracy(y_predict, y_label)
        train = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss_loglikelihood, global_step=global_step)
        # train = tf.train.AdadeltaOptimizer(FLAGS.learning_rate).minimize(loss_loglikelihood,global_step=global_step)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        gstep = 0

        summaries = tf.summary.merge_all()
        # if FLAGS.train:
        #     writer = tf.summary.FileWriter(join(FLAGS.summaries_dir, 'train'),
        #                                    sess.graph)
        # else :
        #     writer = tf.summary.FileWriter(join(FLAGS.summaries_dir, 'test'),
        #                                    sess.graph)

        # train
        if FLAGS.train:
            print('Start training...')
            writer = tf.summary.FileWriter(join(FLAGS.summaries_dir, 'train'),
                                           sess.graph)

            if tf.gfile.Exists(FLAGS.summaries_dir):
                shutil.rmtree(FLAGS.summaries_dir, ignore_errors=True)

            for epoch in range(FLAGS.epoch_num):
                tf.train.global_step(sess, global_step_tensor=global_step)
                # 初始化
                sess.run(train_initializer)
                print(f'Epoch {epoch} initialized.')
                # 每个batch
                for step in range(int(train_steps)):
                    try:
                        # size = sess.run(batchsize)
                        # print('Batchsize',batchsize.eval())
                        # seq_lengths = np.full([size],FLAGS.time_step,dtype = np.int32)
                        smrs, loss, acc, gstep, _ = sess.run([summaries, loss_loglikelihood, accuracy, global_step, train],
                                                         feed_dict={keep_prob: FLAGS.keep_prob})


                        if step % FLAGS.steps_per_print == 0:
                            print('Global Step', gstep, 'Step', step, 'Train Loss', loss, 'Accuracy', acc)
                            # print("y_predict shape, y_label shape", y_predict, y_label)

                        if gstep % FLAGS.steps_per_summary == 0:
                            writer.add_summary(smrs, gstep)
                            print('Writen summaries to', FLAGS.summaries_dir)
                    except tf.errors.OutOfRangeError:
                        print("out of range")
                        break
                    except tf.errors.InvalidArgumentError:
                        print("invalid argument")
                        break

                if epoch % FLAGS.epochs_per_dev == 0:
                    # Dev
                    sess.run(dev_initializer)
                    for step in range(int(dev_steps)):
                        if step % FLAGS.steps_per_print == 0:
                            # size = sess.run(batchsize)
                            # seq_lengths = np.full([size], FLAGS.time_step, dtype=np.int32)
                            try:
                                print('Dev Accuracy', sess.run(accuracy, feed_dict={keep_prob: 1}), 'Step', step)

                            except tf.errors.OutOfRangeError:
                                print("out of range")
                                break

                            except tf.errors.InvalidArgumentError:
                                print("invalid argument")
                                break

                if epoch % FLAGS.epochs_per_save == 0:
                    # 保存模型
                    saver.save(sess, FLAGS.checkpoint_dir, global_step=gstep)
            writer.close()

        # predict
        else:
            print('Start predicting...')
            ckpt = tf.train.get_checkpoint_state('../ckpt')
            # restore model
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restored from', ckpt.model_checkpoint_path)
            # initialize
            # sess.run(test_initializer)
            sess.run(test_iter.initializer)

            # predict
            y_output = []
            # for step in range(int(test_steps)):
            while True:
                try:
                    x_results, y_predict_results = sess.run([x, y_predict], feed_dict={keep_prob: 1})
                    y_output.extend(y_predict_results)
                    # if step % 1000 == 0:
                    #     print('Test Step:',step)

                except tf.errors.OutOfRangeError:
                    print("Out of range!")
                    break

            results = np.array(y_output)
            print('output shape', results.shape)
            results = np.reshape(results, test_x.shape)
            print('reshape output shape', results.shape)
            np.save(outputfile, results)


if __name__ == '__main__':
    main('../data/test_Y.npy')
