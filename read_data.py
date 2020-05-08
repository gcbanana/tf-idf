#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/6 15:37
# @File    : read_data.py


import pandas as pd


SENT_FILE = 'data/ChnSentiCorp_htl_all.csv'
STOP_FILE = 'data/hit_stopwords.txt'


def read_sent_data(file):
    """
    读取文本数据
    :param file:
    :return:
    """
    data = pd.read_csv(file, sep=',')
    return data['review']


def read_stop_data(file):
    """
    读取停用词
    :param file:
    :return:
    """
    stop_words = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            stop_words.append(line.strip())
    return stop_words


if __name__ == '__main__':
    sentences = read_sent_data(SENT_FILE)
    print(sentences)
    stop = read_stop_data(STOP_FILE)
    print(stop)
