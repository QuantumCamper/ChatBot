import os
import json
import random
import numpy as np
import six
import string

import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

import logging

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("min_learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("num_classes", 2, "The output target index.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("mode", "train", "model mode")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("answer_length", 30, "the max answer length.")
tf.app.flags.DEFINE_integer("num_layers", 3, "the num of deep layers.")
tf.app.flags.DEFINE_integer("question_length", 32, "the max question length.")
tf.app.flags.DEFINE_integer("content_length", 450, "the max content length.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def process_data(answer_file_name, content_file_name, question_file_name, target_file):
    answer_file = open(os.path.join(FLAGS.data_dir, answer_file_name), 'rb')
    content_file = open(os.path.join(FLAGS.data_dir, content_file_name), 'rb')
    question_file = open(os.path.join(FLAGS.data_dir, question_file_name), 'rb')
    answer, content, question = answer_file.readline(), content_file.readline(), question_file.readline()
    i = 0
    f = open(os.path.join(FLAGS.data_dir, target_file), 'wb')
    while answer and content:
        content = content.lstrip().rstrip().split(b' ')
        answer = answer.lstrip().rstrip().split(b' ')
        question = question.lstrip().rstrip().split(b' ')
        f.write(b'answer:' + b' '.join(answer) + b',question:' + b' '.join(question) +
                b',content:' + b' '.join(content) + b'\n')
        answer, content, question = answer_file.readline(), content_file.readline(), \
                                    question_file.readline()
        i = i + 1


# 根据给定size对序列进行padding
def pad_input(input_, size, type):
    if type == 'answer':
        out = []
        for item in input_:
            out.append(item + [PAD_ID] * (size - len(item)))
        return out
    else:
        return input_ + [PAD_ID]*(size - len(input_))

def create_masks(batch_size, inputs, length):
    masks = []
    for length_id in range(length):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            if length_id < length - 1:
                target = inputs[length_id + 1][batch_id]

            if length_id == length - 1 or target == PAD_ID:
                batch_mask[batch_id] = 0.0
        masks.append(batch_mask)
    return masks


# 根据给定batch_size对数据集进行batch划分
def _reshape_batch(inputs, size, batch_size):
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append([inputs[batch_id][length_id] for batch_id in range(batch_size)])
    return batch_inputs


def process_answer(answer_input, content):
    labels = []

    start = int(answer_input[0])
    label_start = [0] * len(content)
    label_start[start] = 1
    labels.append(label_start)

    end = int(answer_input[1])
    label_end = [0] * len(content)
    label_end[end] = 1
    labels.append(label_end)
    valid_label = [start, end]
    return labels, valid_label


def get_batch(data, batch_size=1):
    question_inputs, content_inputs, answer_inputs, true_labels = [], [], [], []
    for i in range(batch_size):
        data_batch = random.choice(data)
        answer_input, question_input, content_input = data_batch.strip().split(b',')
        question_input = question_input.strip().split(b':')[1]
        content_input = content_input.strip().split(b':')[1]
        answer_input = answer_input.strip().split(b':')[1]
        question_input = question_input.strip().split(b' ')
        question_input = [int(item) for item in question_input]
        content_input = content_input.strip().split(b' ')
        content_input = [int(item) for item in content_input]
        answer_input = answer_input.strip().split(b' ')

        answer_input, valid_answer = process_answer(answer_input, content_input)

        # pad
        question_inputs.append(pad_input(question_input, FLAGS.question_length, type='question'))
        content_inputs.append(pad_input(content_input, FLAGS.content_length, type='content'))
        answer_inputs.append(pad_input(answer_input, FLAGS.content_length, type='answer'))
        true_labels.append(valid_answer)


    # reshape
    question_inputs = _reshape_batch(question_inputs, FLAGS.question_length, batch_size)
    content_inputs = _reshape_batch(content_inputs, FLAGS.content_length, batch_size)
    answer_inputs = _reshape_batch(answer_inputs, FLAGS.num_classes, batch_size)
    true_labels = _reshape_batch(true_labels, FLAGS.num_classes, batch_size)

    # 按照batch_size进行掩码设计
    question_masks = create_masks(batch_size, question_inputs, FLAGS.question_length)
    content_masks = create_masks(batch_size, content_inputs, FLAGS.content_length)
    inputs = [question_inputs, content_inputs, question_masks, content_masks]
    labels = answer_inputs
    return inputs, labels, true_labels


def main(_):
    f = open(os.path.join(FLAGS.data_dir, 'train.data'), 'rb')
    data = f.readlines()
    num_total = len(data)
    train_epoch = num_total//FLAGS.epochs
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)
    max_seq = (FLAGS.question_length, FLAGS.content_length)
    f = np.load(embed_path)
    glove = f['glove'].astype(np.float32)

    qa = QASystem(encoder, decoder, max_seq, glove, FLAGS, 'train')

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        # load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        load_train_dir = FLAGS.train_dir
        initialize_model(sess, qa, load_train_dir)

        # save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        save_train_dir = FLAGS.train_dir
        for i in range(0, train_epoch):
            inputs, labels, _ = get_batch(data, batch_size=FLAGS.batch_size)
            qa.train(sess, inputs, labels, i, FLAGS.print_every, save_train_dir)
            qa.saver.save(sess, save_path=FLAGS.train_dir + '/match_lstm', global_step=i)
            valid_inputs, valid_labels, _ = get_batch(data, batch_size=FLAGS.batch_size)
            if i % 100 == 0:
                qa.validate(sess, valid_inputs, valid_labels, i)


def eval():
    f2 = open(os.path.join(FLAGS.data_dir, 'val.data'), 'rb')
    data = f2.readlines()
    num_total = len(data)
    eval_epoch = 100 // FLAGS.epochs
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)
    max_seq = (FLAGS.question_length, FLAGS.content_length)
    f = np.load(embed_path)
    glove = f['glove'].astype(np.float32)

    qa = QASystem(encoder, decoder, max_seq, glove, FLAGS, 'answer')

    with tf.Session() as sess:
        load_train_dir = FLAGS.train_dir
        initialize_model(sess, qa, load_train_dir)
        for i in range(0, eval_epoch):
            inputs, _, target = get_batch(data, batch_size=FLAGS.batch_size)
            qa.evaluate_answer(sess, inputs, target, vocab)


if __name__ == "__main__":
    tf.app.run()
    # process_data('val.span', 'val.ids.context', 'val.ids.question', 'val.data')




