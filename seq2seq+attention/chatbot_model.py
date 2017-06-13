#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import inspect
from tensorflow.contrib import rnn, seq2seq
from data_preprocessing import Config
import time


class ChatBotModel(object):
    def __init__(self, forward_only, batch_size):
        print("Initialize Model...")
        self.forward_only = forward_only  # 训练完后forward_only设为True，测试不再进行反向传播
        self.batch_size = batch_size
        with tf.variable_scope('training') as scope:
            self.global_steps = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_steps')

    def _create_placeholders(self):
        print("create placeholders")
        self.encode_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                              for i in range(Config.BUCKETS[-1][0])]
        self.decode_inputs = [tf.placeholder(tf.int32, shape=[None], name='decode{}'.format(i))
                              for i in range(Config.BUCKETS[-1][1] + 1)]
        self.decode_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                             for i in range(Config.BUCKETS[-1][1] + 1)]
        # target为decode输入平移一位
        self.targets = self.decode_inputs[1:]

    def _create_inference(self):
        print("Create Inference")
        # 设置decode词嵌入变换矩阵，decode输出的词作为下一step输入时，使用该矩阵映射到具体的词
        if (Config.NUM_SAMPLES > 0) and (Config.NUM_SAMPLES < Config.DEC_VOCAB):
            with tf.variable_scope('output_projection') as scope:
                w = tf.get_variable('output_projection_w', shape=[Config.HIDDEN_SIZE, Config.DEC_VOCAB],
                                    dtype=tf.float32)
                b = tf.get_variable('output_projection_b', shape=[Config.DEC_VOCAB], dtype=tf.float32)
                self.output_projection = (w, b)

        def sample_loss(inputs, labels):
            inputs = tf.reshape(inputs, shape=[-1, 1])
            return tf.nn.sampled_softmax_loss(weights=tf.transpose(w), biases=b,
                                              labels=inputs, inputs=labels, num_sampled=Config.NUM_SAMPLES,
                                              num_classes=Config.DEC_VOCAB)

        self.softmax_loss_function = sample_loss
        # 采用GRU CELL
        single_cell = rnn.GRUCell(Config.HIDDEN_SIZE)
        self.cell = rnn.MultiRNNCell([single_cell for _ in range(Config.NUM_LAYERS)])

    def _create_loss(self):
        print('Create Loss')
        start = time.time()

        # 定义所使用的seq2seq函数
        def seq2seq_function(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs, cell=self.cell,
                num_encoder_symbols=Config.ENC_VOCAB,
                num_decoder_symbols=Config.DEC_VOCAB,
                embedding_size=Config.HIDDEN_SIZE,
                output_projection=self.output_projection,
                feed_previous=do_decode
            )
        # 需要对训练和预测过程分别进行定义
        if self.forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                encoder_inputs=self.encode_inputs,
                decoder_inputs=self.decode_inputs,
                targets=self.targets,
                weights=self.decode_masks,
                buckets=Config.BUCKETS,
                seq2seq=lambda x, y: seq2seq_function(x, y, True),
                softmax_loss_function=self.softmax_loss_function)

            # 由于采用sampling softmax, 需要对每个BUCKET分别计算解码
            if self.output_projection:
                for bucket in range(len(Config.BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output,
                                            self.output_projection[0]) + self.output_projection[1]
                                            for output in self.outputs[bucket]]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                encoder_inputs=self.encode_inputs,
                decoder_inputs=self.decode_inputs,
                targets=self.targets,
                weights=self.decode_masks,
                buckets=Config.BUCKETS,
                seq2seq=lambda x, y: seq2seq_function(x, y, False),
                softmax_loss_function=self.softmax_loss_function)
        print('Time:', time.time() - start)

    def _create_optimizer(self):
        print('Create Optimizer')
        with tf.variable_scope('training') as scope:
            if not self.forward_only:
                self.optimizer = tf.train.GradientDescentOptimizer(Config.LR)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                start = time.time()
                for bucket in range(len(Config.BUCKETS)):
                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket], trainables),
                                                                 clip_norm=Config.MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables),
                                                                         global_step=self.global_steps))
                    print('Creating opt for bucket {} took {} seconds'.format(bucket, time.time() - start))
                    start = time.time()

    def _create_summary(self):
        pass

    def build_graphs(self):
        self._create_placeholders()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()













