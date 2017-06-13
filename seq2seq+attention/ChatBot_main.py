#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import random
import numpy as np
import time
import tensorflow as tf
import os
import sys
import data_preprocessing
from data_preprocessing import Config
from chatbot_model import ChatBotModel

# 加载buckets数据
def get_buckets():
    # 加载数据
    test_buckets = data_preprocessing.load_data('test_ids.encode', 'test_ids.decode')
    train_buckets = data_preprocessing.load_data('train_ids.encode', 'train_ids.decode')
    train_buckets_sizes = [len(train_buckets[b]) for b in range(len(Config.BUCKETS))]
    print('buckets sizes：', train_buckets_sizes)
    total_train_buckets = sum(train_buckets_sizes)
    # 为便于选择buckets，将其size规范化为0-1数值
    train_buckets_scale = [sum(train_buckets_sizes[:i+1])/total_train_buckets for i in range(len(train_buckets_sizes))]
    return test_buckets, train_buckets, train_buckets_scale


def get_random_bucket(train_buckets_scale):
    rand = random.random()
    return min(i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > rand)


# 测试输入长度与模型设定是否匹配
def assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket, %d != %d."
                         % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket, %d != %d."
                         % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket, %d != %d."
                         % (len(decoder_masks), decoder_size))


# session执行迭代
def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    encoder_size, decoder_size = Config.BUCKETS[bucket_id]
    assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)
    # session feed placeholders
    feed_dict = {}
    for step in range(encoder_size):
        feed_dict[model.encode_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        feed_dict[model.decode_inputs[step].name] = decoder_inputs[step]
        feed_dict[model.decode_masks[step].name] = decoder_masks[step]
    last_target = model.decode_inputs[decoder_size].name
    feed_dict[last_target] = np.zeros([model.batch_size], np.int32)

    # session output
    if not forward_only:
        output_feed = [model.train_ops[bucket_id], model.gradient_norms[bucket_id], model.losses[bucket_id]]
    else:
        output_feed = [model.losses[bucket_id]]
        for step in range(decoder_size):
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, feed_dict)
    if not forward_only:
        return outputs[1], outputs[2], None
    else:
        return None, outputs[0], outputs[1:]


# 断点restore
def _checkpoints_restore(sess, saver):
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(Config.CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print('Restore Parameters')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the Chatbot")


class Train:
    def __init__(self):
        self.test_buckets, self.train_buckets, self.train_buckets_scale = get_buckets()

    # 设置每多少次存储一次断点
    def _get_skip_step(self, iteration):
        if iteration < 100:
            return 30
        return 100

    # 在测试集上进行评估
    def _eval_test_set(self, sess, model, test_buckets):
        """ Evaluate on the test set. """
        for bucket_id in range(len(Config.BUCKETS)):
            if len(test_buckets[bucket_id]) == 0:
                print("  Test: empty bucket %d" % (bucket_id))
                continue
            start = time.time()
            encoder_inputs, decoder_inputs, decoder_masks = data_preprocessing.get_batch(test_buckets[bucket_id],
                                                                                         bucket_id,
                                                                                         batch_size=Config.BATCH_SIZE)
            _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs,
                                       decoder_masks, bucket_id, True)
            print('Test bucket {}: loss {}, time {}'.format(bucket_id, step_loss, time.time() - start))

    def train(self):
        model = ChatBotModel(False, Config.BATCH_SIZE)
        model.build_graphs()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('Running Session')
            sess.run(tf.global_variables_initializer())
            _checkpoints_restore(sess, saver)
            iteration = model.global_steps.eval()
            total_loss = 0.0

            while True:
                skip_step = self._get_skip_step(iteration)
                bucket_id = get_random_bucket(self.train_buckets_scale)
                encoder_inputs, decoder_inputs, decoder_masks = data_preprocessing.get_batch(
                    self.train_buckets[bucket_id], bucket_id, batch_size=Config.BATCH_SIZE)

                start = time.time()
                _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, False)
                total_loss += step_loss
                iteration += 1
                if iteration % skip_step == 0:
                    print('iteration{}:loss{},time{}'.format(iteration, total_loss, time.time()-start))
                    start = time.time()
                    total_loss = 0
                    saver.save(sess=sess, save_path=Config.CPT_PATH+'/bot', global_step=iteration)
                    if iteration % (10*skip_step):
                        self._eval_test_set(sess, model, self.test_buckets)
                        start = time.time()
                    sys.stdout.flush()

# chat对话输出
class Chat:
    def __init__(self):
        # 加载字典
        _, self.encode_vocab = data_preprocessing.load_vocab(os.path.join(Config.PROCESSED_PATH, 'vocab.encode'))
        self.decode_vocab, _ = data_preprocessing.load_vocab(os.path.join(Config.PROCESSED_PATH, 'vocab.decode'))

    def _get_user_input(self):
        print("> ", end="")
        sys.stdout.flush()
        return sys.stdin.readline()

    def _find_right_bucket(self, length):
        return min([b for b in range(len(Config.BUCKETS))
                    if Config.BUCKETS[b][0] >= length])

    # 根据贪婪算法获取当前概率最大的词
    def _construct_response(self, output_logits, inv_dec_vocab):
        # print(output_logits[0])
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # 切除最后的EOS标记
        if Config.EOS_ID in outputs:
            outputs = outputs[:outputs.index(Config.EOS_ID)]
        # print(outputs)
        return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])

    def chat(self):
        model = ChatBotModel(True, batch_size=1)
        model.build_graphs()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            _checkpoints_restore(sess, saver)
            max_length = Config.BUCKETS[-1][0]
            output_file = open(os.path.join(Config.PROCESSED_PATH, Config.OUTPUT_FILE), 'a+')
            print('Welcome to TensorBOT. Say something. Enter to exit. Max length is', max_length)

            while True:
                line = self._get_user_input()
                if len(line) > 0 and line[-1] == '\n':
                    line = line[:-1]
                if line == '':
                    break
                output_file.write('HUMAN ++++ ' + line + '\n')
                # 将输入转换为token_id
                token_ids = data_preprocessing.sentence2id(self.encode_vocab, str(line))
                if len(token_ids) > max_length:
                    print('Max length I can handle is:', max_length)
                    line = self._get_user_input()
                    continue

                # 查找用户输入话语长度对应的bucket
                bucket_id = self._find_right_bucket(len(token_ids))
                # 用户输入话语feed到模型中
                encoder_inputs, decoder_inputs, decoder_masks = data_preprocessing.get_batch([(token_ids, [])],
                                                                                             bucket_id, batch_size=1)
                _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                               decoder_masks, bucket_id, True)
                response = self._construct_response(output_logits, self.decode_vocab)
                print(response)
                output_file.write('BOT ++++ ' + response + '\n')
            output_file.write('=============================================\n')
            output_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat'}, default='chat', help='Please Set The Mode')
    args = parser.parse_args()
    # 准备训练数据
    if not os.path.isdir(Config.PROCESSED_PATH):
        data_preprocessing.raw_data_process()
        data_preprocessing.process_data()
    print('Data Ready')
    # 创建断点存储文件
    data_preprocessing.make_dir(os.path.join(Config.DATA_PATH, Config.CPT_PATH))

    if args.mode == 'train':
        train_mode = Train()
        train_mode.train()
    elif args.mode == 'chat':
        test_mode = Chat()
        test_mode.chat()

if __name__ == "__main__":
    main()

