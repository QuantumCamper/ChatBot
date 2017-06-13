#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import re
import numpy as np


class Config:
    DATA_PATH = 'cornell movie-dialogs corpus/'
    LINE_FILE = 'movie_lines.txt'
    CONVO_FILE = 'movie_conversations.txt'
    PROCESSED = 'processed'
    PROCESSED_PATH = 'cornell movie-dialogs corpus/processed'
    CPT_PATH = 'checkpoints'
    OUTPUT_FILE = 'output_convo.txt'
    TEST_SIZE = 25000
    BATCH_SIZE = 64
    THRESHOLD = 2
    BUCKETS = [(8, 10), (12, 14), (16, 19)]
    PAD_ID = 0
    HIDDEN_SIZE = 256
    DEC_VOCAB = 18760
    ENC_VOCAB = 18603
    NUM_SAMPLES = 512
    NUM_LAYERS = 3
    LR = 0.5
    MAX_GRAD_NORM = 5.0
    EOS_ID = 3


# 清洗movie_lines.txt
def get_lines():
    id2line = {}
    file_path = os.path.join(Config.DATA_PATH, Config.LINE_FILE)
    with open(file_path, 'rb') as f:
        for line in f.readlines():
            parts = line.split(' +++$+++ '.encode('utf-8'))
            # 去除‘\n'
            if len(parts) == 5:
                if parts[4][-1] == '\n':
                    parts[4] = parts[4][:-1]
                # 以line编号为键值
                id2line[parts[0]] = parts[4]
    return id2line


# 清洗movie_lines.txt
def get_convos():
    file_path = os.path.join(Config.DATA_PATH, Config.CONVO_FILE)
    conversations = []
    with open(file_path, 'rb') as f:
        for line in f.readlines():
            parts = line.split(' +++$+++ '.encode('utf-8'))
            if len(parts) == 4:
                convo = []
                for part in parts[3][1:-2].split(', '.encode('utf-8')):
                    part = part.split('\''.encode('utf-8'))[1]
                    convo.append(part)
            conversations.append(convo)
    return conversations


# 将数据集划分为questions and answers两个集合
def questions_answers(id2line, conversations):
    questions, answers = [], []
    for convos in conversations:
        for index, line in enumerate(convos[:-1]):
            questions.append(id2line[convos[index]])
            answers.append(id2line[convos[index + 1]])
    assert len(questions) == len(answers)
    return questions, answers


# 划分数据集train,test
def prepare_dataset(questions, answers, processed_path):
    filenames = ['train.encode', 'train.decode', 'test.encode', 'test.decode']
    test_ids = random.sample([i for i in range(len(questions))], Config.TEST_SIZE)
    files = []
    for file_name in filenames:
        files.append(open(os.path.join(processed_path, file_name), 'wb'))
    for i in range(len(questions)):
        if i in test_ids:
            files[2].write(questions[i])
            files[3].write(answers[i])
        else:
            files[0].write(questions[i])
            files[1].write(answers[i])
    for file in files:
        file.close()

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def raw_data_process():
    make_dir(Config.PROCESSED_PATH)
    id2line = get_lines()
    conversations = get_convos()
    questions, answers = questions_answers(id2line, conversations)
    prepare_dataset(questions, answers, Config.PROCESSED_PATH)


# 构建分词器
def tokenizer(line, normalize_digits=True):
    line = line.encode()
    line = re.sub(b'<u>', b'', line)
    line = re.sub(b'</u>', b'', line)
    line = re.sub(b'\[', b'', line)
    line = re.sub(b'\]', b'', line)
    words = []
    pattern = re.compile(b"([.,!?\"'-<>:;)(])")
    digit_re = re.compile(b"\d")
    for segment in line.strip().lower().split():
        for token in re.split(pattern, segment):
            if not token:
                continue
            if normalize_digits:
                re.sub(digit_re, b'#', token)
            words.append(token)
    return words


# 构建字典
def build_vocab(filename, normalize_digits=True):
    processed_path = os.path.join(Config.DATA_PATH, Config.PROCESSED)
    read_path = os.path.join(processed_path, filename)
    out_path = os.path.join(processed_path, 'vocab.{}'.format(filename[-6:]))
    vocab = {}
    with open(read_path, 'rb') as f:
        for line in f.readlines():
            for token in tokenizer(line):
                if token not in vocab:
                    vocab[token] = 0
                else:
                    vocab[token] += 1
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'wb') as f:
        f.write(b'<pad>' + b'\n')
        f.write(b'<unk>' + b'\n')
        f.write(b'<s>' + b'\n')
        f.write(b'<\s>' + b'\n')
        index = 4
        for word in sorted_vocab:
            if vocab[word] < Config.THRESHOLD:
                with open('config.py', 'ab') as cf:
                    if filename[-6:] == 'encode':
                        cf.write(b'ENC_VOCAB = ' + bytes(index) + b'\n')
                    else:
                        cf.write(b'DEC_VOCAB = ' + bytes(index) + b'\n')
                break
            f.write(word + b'\n')
            index += 1


# 下载字典方法
def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}


# 句子解析id
def sentence2id(vocab, line):
    # 如果字典中不存在就返回<unk>
    return [vocab.get(token, vocab[b'<unk>']) for token in tokenizer(line)]


# token解析id
def token2id(data, mode):
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(Config.PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(Config.PROCESSED_PATH, in_path), 'rb')
    out_file = open(os.path.join(Config.PROCESSED_PATH, out_path), 'wb')

    lines = in_file.read().splitlines()
    for line in lines:
        if mode == 'decode':
            ids = [vocab[b'<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        if mode == 'decode':
            ids.append(vocab[b'<\s>'])
        out_file.write((' '.join(str(id_) for id_ in ids) + '\n').encode("utf-8"))


def process_data():
    print('Preparing data to be model-ready ...')
    build_vocab('train.encode')
    build_vocab('train.decode')
    token2id('train', 'encode')
    token2id('train', 'decode')
    token2id('test', 'encode')
    token2id('test', 'decode')


# 构建buckets
def load_data(enc_filename, dec_filename, max_training_size=None):
    encode_file = open(os.path.join(Config.PROCESSED_PATH, enc_filename), 'rb')
    decode_file = open(os.path.join(Config.PROCESSED_PATH, dec_filename), 'rb')
    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in Config.BUCKETS]
    i = 0
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)
        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(Config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_buckets


# 根据给定size对序列进行padding
def pad_input(input_, size):
    return input_ + [Config.PAD_ID]*(size - len(input_))


# 根据给定batch_size对数据集进行batch划分
def _reshape_batch(inputs, size, batch_size):
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                      for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


# 对给定bucket_id对应的bucket进行padding，并获取相应的batch_size个样本
def get_batch(data_bucket, bucket_id, batch_size=1):
    encode_max_size, decode_max_size = Config.BUCKETS[bucket_id]
    encode_inputs, decode_inputs = [], []
    for i in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # encode端的输入需要进行顺序反转
        encode_inputs.append(list(reversed(pad_input(encoder_input, encode_max_size))))
        decode_inputs.append(pad_input(decoder_input, decode_max_size))

    # 据给定batch_size对数据集进行batch划分
    batch_encoder_inputs = _reshape_batch(encode_inputs, encode_max_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decode_inputs, decode_max_size, batch_size)

    # 按照batch_size对decode进行掩码设计
    batch_masks = []
    for length_id in range(decode_max_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            if length_id < decode_max_size - 1:
                target = decode_inputs[batch_id][length_id + 1]

            if length_id == decode_max_size - 1 or target == Config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks

if __name__ == "__main__":
    # raw_data_process()
    # process_data()
    # data_buckets = load_data('train_ids.encode', 'train_ids.decode')
    # batch_encoder_inputs, batch_decoder_inputs, batch_masks = get_batch(data_buckets[0], 0)
    # print(batch_encoder_inputs)
    # print(batch_masks)
    f = open(os.path.join(Config.PROCESSED_PATH, 'vocab.encode'), 'rb')
    length = 0
    for line in f.readlines():
        if len(line) > 0:
            length += 1
    print(length)










