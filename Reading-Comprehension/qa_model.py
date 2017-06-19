import time
import logging

import numpy as np
from seq2seq_modify import attention_decoder, _extract_argmax_and_embed
import tensorflow as tf
from tensorflow.contrib import rnn
from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt, learning_rate):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer(learning_rate)
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, batch_size, encoder_max_size, num_layers):
        length = tf.cast(tf.reduce_sum(masks, axis=0), tf.int32)
        cell_fw = rnn.LSTMCell(num_units=self.size,
                               initializer=tf.random_uniform_initializer(-0.1, 0.1),
                               state_is_tuple=False)
        cell_fw = rnn.MultiRNNCell([cell_fw]*num_layers, state_is_tuple=False)
        cell_bw = rnn.LSTMCell(num_units=self.size,
                               initializer=tf.random_uniform_initializer(-0.1, 0.1),
                               state_is_tuple=False)
        cell_bw = rnn.MultiRNNCell([cell_bw] * num_layers, state_is_tuple=False)

        outputs, outputs_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                   cell_bw=cell_bw,
                                                                   inputs=inputs,
                                                                   sequence_length=length,
                                                                   dtype=tf.float32,
                                                                   time_major=True)

        encode_outputs = tf.concat(outputs, 2)
        encode_outputs = tf.reshape(encode_outputs, shape=[batch_size, encoder_max_size, 2 * self.size])

        return encode_outputs


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep, decoder_inputs, embeddings, batch_size, loop_function=None):

        cell = rnn.LSTMCell(num_units=self.output_size, initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                state_is_tuple=False)
        decoder_inputs = tf.unstack(decoder_inputs, axis=0)
        init_status = tf.truncated_normal(shape=[batch_size, 2*self.output_size])
        outputs, states, attention_marks = attention_decoder(decoder_inputs=decoder_inputs,
                                                                 initial_state=init_status,
                                                                 attention_states=knowledge_rep,
                                                                 cell=cell, output_size=self.output_size,
                                                                 initial_state_attention=loop_function)

        return outputs, states, attention_marks


class QASystem(object):
    def __init__(self, encoder, decoder, max_seq, dictionary, flags, mode):
        """
        Initializes System
        """
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_max_size, self.decoder_max_size = max_seq
        self.embeddings = dictionary
        self.flags = flags
        self.global_steps = tf.Variable(0, trainable=False, name='global_steps')
        self.total_loss = 0.0
        self.mode = mode

        # ==== set up placeholder tokens ========
        with tf.name_scope('placeholder'):
            self.encoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[self.flags.batch_size],
                                                  name='encoder{}'.
                                                  format(i)) for i in range(self.encoder_max_size)]
            self.decoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[self.flags.batch_size],
                                                  name='decoder{}'.
                                                  format(i)) for i in range(self.decoder_max_size)]
            self.encoder_masks = [tf.placeholder(dtype=tf.float32, shape=[self.flags.batch_size],
                                                 name='encoder_mask{}'.format(i))
                                  for i in range(self.encoder_max_size)]
            self.decoder_masks = [tf.placeholder(dtype=tf.float32, shape=[self.flags.batch_size],
                                                 name='decoder_mask{}'.format(i))
                                  for i in range(self.decoder_max_size)]
            self.answer = [tf.placeholder(dtype=tf.float32, shape=[self.flags.batch_size,
                                                                 self.decoder_max_size],
                                                 name='answer{}'.format(i))
                                  for i in range(self.flags.num_classes)]

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings(self.embeddings)
            self.setup_system()
            self.setup_loss(self.decoder_masks)
            self.setup_optimizer()
            self.summary = tf.summary.merge_all()
            self.saver = tf.train.Saver()

    def setup_system(self):
        """
        After  modularized implementation of encoder and decoder
        call various functions inside encoder, decoder here
        to assemble reading comprehension system!
        """
        encoder_embed, decoder_embed = self.setup_embeddings(self.embeddings)
        with tf.variable_scope('encoder'):
            encode_outputs = self.encoder.encode(encoder_embed, self.encoder_masks,
                                                 self.flags.batch_size,
                                                 self.encoder_max_size, self.flags.num_layers)
        with tf.variable_scope('decoder'):
            outputs, outputs_states, attention_marks = self.decoder.decode(encode_outputs, decoder_embed,
                                                                           self.embeddings,
                                                                           self.flags.batch_size)

        # create a mixture of question and paragraph representation
        encode_rep = []
        for item in attention_marks:
            item = tf.expand_dims(item, axis=1)
            rep = tf.matmul(item, encode_outputs)
            rep = tf.unstack(rep, axis=1)[0]
            encode_rep.append(rep)
        z_mix = tf.concat([outputs, encode_rep], 2)

        with tf.variable_scope('decoder_output_projection'):
            w = tf.get_variable('dec_weights', shape=[self.flags.output_size, self.decoder_max_size],
                                dtype=tf.float32)
            b = tf.get_variable('dec_bias', shape=[self.decoder_max_size], dtype=tf.float32)
            self.output_projection = (w, b)


        # match BILSTM
        with tf.variable_scope('match_bilstm'):
            match_outputs = self.encoder.encode(z_mix, self.decoder_masks, self.flags.batch_size,
                                                   self.decoder_max_size, self.flags.num_layers)
            if self.mode == 'answer':
                loop_function = self._extract_argmax_and_embed(self.output_projection)
                self.answer_outputs, _, _ = self.decoder.decode(match_outputs, self.answer,
                                                                self.embeddings,
                                                                self.flags.batch_size,
                                                                loop_function=loop_function)
            else:
                self.answer_outputs, _, _ = self.decoder.decode(match_outputs, self.answer,
                                                                self.embeddings,
                                                                self.flags.batch_size)

    def setup_loss(self, decoder_masks):
        """
        Set up loss computation
        """
        w = self.output_projection[0]
        b = self.output_projection[1]
        self.answer_outputs = tf.unstack(self.answer_outputs, axis=0)
        logits = []
        for item in self.answer_outputs:
            logits.append(tf.nn.xw_plus_b(x=item, weights=w, biases=b))
        self.logits = logits
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                              labels=self.answer))
            tf.summary.scalar('loss', self.loss)

    def setup_optimizer(self):
        self.lr = tf.maximum(self.flags.min_learning_rate, tf.train.exponential_decay(
            self.flags.learning_rate, self.global_steps, decay_steps=10000, decay_rate=0.98))
        tf.summary.scalar('learning_rate', self.lr)
        optimizer = get_optimizer(self.flags.optimizer, self.lr)
        params = tf.trainable_variables()
        grads, global_norms = tf.clip_by_global_norm(tf.gradients(self.loss, params),
                                                   clip_norm=self.flags.max_gradient_norm)
        tf.summary.scalar('global_norms', global_norms)
        self.optimizer = optimizer.apply_gradients(zip(grads, params), global_step=self.global_steps)

    def setup_embeddings(self, dictionary):
        """
        Loads distributed word representations based on placeholder tokens
        """
        with tf.variable_scope("embeddings"), tf.device('/cpu:0'):
            encoder_embed = tf.nn.embedding_lookup(dictionary, ids=self.encoder_inputs)
            decoder_embed = tf.nn.embedding_lookup(dictionary, ids=self.decoder_inputs)
        return encoder_embed, decoder_embed

    def optimize(self, session, train_x, train_y):
        feed_dict = {}
        for step in range(self.encoder_max_size):
            feed_dict[self.encoder_inputs[step].name] = train_x[0][step]
            feed_dict[self.encoder_masks[step].name] = train_x[2][step]

        for step in range(self.decoder_max_size):
            feed_dict[self.decoder_inputs[step].name] = train_x[1][step]
            feed_dict[self.decoder_masks[step].name] = train_x[3][step]
        for i in range(self.flags.num_classes):
            feed_dict[self.answer[i].name] = train_y[i]

        output_feed = [self.summary, self.optimizer, self.loss, self.logits]

        outputs = session.run(output_feed, feed_dict)

        return outputs

    def _extract_argmax_and_embed(self,output_projection):
        def loop_function(prev, _):
            if output_projection is not None:
                prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
            prev_symbol = tf.argmax(prev, 1)
            # Note that gradients will not propagate through the second parameter of
            # embedding_lookup.
            emb_prev = tf.zeros(prev)
            emb_prev[prev_symbol] = 1
            return emb_prev

        return loop_function

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        feed_dict = {}
        for step in range(self.encoder_max_size):
            feed_dict[self.encoder_inputs[step].name] = valid_x[0][step]
            feed_dict[self.encoder_masks[step].name] = valid_x[2][step]

        for step in range(self.decoder_max_size):
            feed_dict[self.decoder_inputs[step].name] = valid_x[1][step]
            feed_dict[self.decoder_masks[step].name] = valid_x[3][step]
        for i in range(self.flags.num_classes):
            feed_dict[self.answer[i].name] = valid_y[i]

        output_feed = [self.summary, self.loss]

        _, outputs = session.run(output_feed, feed_dict)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        feed_dict = {}
        for step in range(self.encoder_max_size):
            feed_dict[self.encoder_inputs[step].name] = test_x[0][step]
            feed_dict[self.encoder_masks[step].name] = test_x[2][step]

        for step in range(self.decoder_max_size):
            feed_dict[self.decoder_inputs[step].name] = test_x[1][step]
            feed_dict[self.decoder_masks[step].name] = test_x[3][step]
        for i in range(self.flags.num_classes):
            feed_dict[self.answer[i].name] = np.zeros(shape=[self.flags.batch_size, self.decoder_max_size])

        output_feed = [self.logits]
        outputs = session.run(output_feed, feed_dict)
        outputs = np.squeeze(outputs, axis=0)
        yp = outputs[0]
        yp2 = outputs[1]
        return yp, yp2

    def answer_eval(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_inputs, valid_labels, iter):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        """
        valid_cost = self.test(sess, valid_inputs, valid_labels)
        print('iter:{}, valid_cost:{}'.format(iter, valid_cost))
        return valid_cost

    def evaluate_answer(self, session, inputs, target_set, vocab, log=False, sample=100):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels
        """
        f1 = 0.
        em = 0.
        # 字典解析
        value_dict = []
        key_dic = []
        for key, value in vocab.items():
            value_dict.append(value)
            key_dic.append(key)

        content = inputs[1]
        target = target_set
        answer = self.answer_eval(session, inputs)

        # 真实answer和预测answer
        for i in range(self.flags.batch_size):
            start = answer[0][i]
            end = answer[1][i]
            target_start = target[0][i]
            target_end = target[1][i]
            predict_seq = []
            target_seq = []
            # 预测answer
            for loc in range(int(start), int(end)+1):
                voc_index = content[loc][i]
                word_index = value_dict.index(voc_index)
                word = key_dic[word_index]
                predict_seq.append(word)
            # 真实answer
            for loc in range(int(target_start), int(target_end)+1):
                voc_index = content[loc][i]
                word_index = value_dict.index(voc_index)
                word = key_dic[word_index]
                target_seq.append(word)

            # 计算F1 and Exact Match
            predict_seq = ' '.join(predict_seq)
            target_seq = ' '.join(target_seq)
            f1 += f1_score(predict_seq, target_seq)
            em += exact_match_score(predict_seq, target_seq)

        # 计算平均值
        f1 = f1/sample
        em = em/sample
        print('f1:{},em:{}'.format(f1, em))

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, inputs, labels,  epochs, print_every, train_dir):
        """
        Implement main training loop

        """
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        writer = tf.summary.FileWriter('.train/summary', graph=session.graph)
        summaries, _, batch_loss, _ = self.optimize(session, inputs, labels)
        self.total_loss += batch_loss
        writer.add_summary(summaries, global_step=epochs)
        if epochs % print_every == 0:
            print('epochs{}, loss:{}, time:{}'.format(epochs, self.total_loss, time.time()-tic))
            self.total_loss = 0








