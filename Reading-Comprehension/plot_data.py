#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
len_question = []
len_answer = []
len_content = []
f_answer = open('./data/squad/train.answer', 'rb')
for line in f_answer.readlines():
    line = line[:-2].split(b' ')
    len_answer.append(len(line))
f_answer.close()

f_question = open('./data/squad/train.question', 'rb')
for line in f_question.readlines():
    line = line[:-2].split(b' ')
    len_question.append(len(line))
f_question.close()

f_content = open('./data/squad/train.context', 'rb')
for line in f_content.readlines():
    line = line[:-2].split(b' ')
    len_content.append(len(line))
f_content.close()

assert len(len_answer) == len(len_question)
assert len(len_answer) == len(len_content)
hist_plot = [len_answer, len_question, len_content]
plt_title = ['answer', 'question', 'content']
plt.figure(facecolor='w')
for i, title in enumerate(plt_title):
    plt.subplot(1, 3, i+1)
    weights = np.ones_like(hist_plot[i])/float(len(hist_plot[i]))
    plt.hist(hist_plot[i], weights=weights)
    plt.title(title)
plt.show()







