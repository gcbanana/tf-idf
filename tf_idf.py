#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/6 16:13
# @File    : tf_idf.py


import math
import jieba
import pandas as pd
from collections import OrderedDict
from read_data import read_sent_data, read_stop_data, SENT_FILE, STOP_FILE


class TfIdf(object):

    def __init__(self):
        self.docs = None  # pandas obj
        self.stop_words = []
        # [['盐城', '最低气温', '晚上', '冷得', '要死', '居然', '不开', '空调', '投诉',
        # '酒店客房', '得到', '答复', '现在', '没有', '领导', '指示', '需要', '暖气', '冷到',
        # '一床', '被子', '可怜'], ['实在', '失望', '之前', '其他人', '点评', '觉得', '大跌眼镜',
        # '这家', '酒店', '以后', '无论如何', '不会']]
        self.seg_docs = self.get_seg_docs()
        self.tf = []  # 每个句子中词频数/词总数 [{word1:0.3,word2:0.4,word4:0.3},{word2:0.5,word3:0.3,word4:0.2},{...},...]
        self.df = {}  # 每个词出现了多少个文档中 {word1:6個文檔,word2:3個文檔,word3:5個文檔,word4:4個文檔...}
        self.idf = {}  # 通过df和文档总数计算得来的idf，可以过滤掉出现次数太少的，也可以不过滤，防止分母溢出
        self.top_k_idf = {}  # top k 大的idf
        self.bow = {}  # 最终文本转换向量的字典
        self.cal_tfidf()

    def get_seg_docs(self):
        """
        处理文本，过滤停用词
        :return:
        """
        seg_docs = []
        self.stop_words = read_stop_data(STOP_FILE)
        self.docs = read_sent_data(SENT_FILE)
        for sent in self.docs:
            if not pd.isnull(sent):
                seg_sent = [w for w in jieba.lcut(sent) if w not in self.stop_words and w.isalpha()]
                seg_docs.append(seg_sent)
        return seg_docs

    def cal_tfidf(self):
        """
        计算tf df idf
        :return:
        """
        for doc in self.seg_docs:
            bow = {}
            for word in doc:
                if word not in bow:
                    bow[word] = 1
                else:
                    bow[word] += 1
            for word in bow.keys():
                # 归一，防止它偏向长的文件（同一个词语在长文件里可能会比短文件有更高的词频，而不管该词语重要与否）
                bow[word] = bow[word] / len(doc)
            self.tf.append(bow)

            for word in bow.keys():
                if word not in self.df:
                    self.df[word] = 1
                else:
                    self.df[word] += 1

        for word, df in self.df.items():
            if df < 10:
                pass
            else:
                self.idf[word] = math.log10(len(self.seg_docs) / df)
            # self.idf[word] = math.log10(len(self.seg_docs) / (df + 1))

    def tf(self, index, word):
        """
        获取第index句的word的tf
        :param index:
        :param word:
        :return:
        """
        return self.tf[index][word]

    def idf(self, word):
        """
        获取word的idf
        :param word:
        :return:
        """
        return self.idf[word]

    def tf_idf(self, index, word):
        """
        获取第index文档的word的tf-idf
        :param index:
        :param word:
        :return:
        """
        return self.tf[index][word] * self.idf[word]

    def get_top_k_idf(self, k, reverse=True):
        """
        获取前k大idf的word
        :param k:
        :param reverse:
        :return:
        """
        self.top_k_idf = OrderedDict(sorted(self.idf.items(), key=lambda x: x[1], reverse=reverse)[:k])
        return self.top_k_idf

    def set_bag_of_word(self, bow):
        """
        设置最终文本转化为向量的词表
        :param bow:
        :return:
        """
        self.bow = bow

    def get_text_vector(self, index):
        """
        文本转向量
        :return:
        """
        return [self.tf_idf(index, w) if w in self.seg_docs[index] else 0 for w in self.bow]

    @staticmethod
    def cosine_similarity(v1, v2):
        """
        计算cosine相似度
        :param v1:
        :param v2:
        :return:
        """
        sum_xx = 0.0
        sum_xy = 0.0
        sum_yy = 0.0

        for i in range(len(v1)):
            x, y = v1[i], v2[i]
            sum_xx += math.pow(x, 2)
            sum_yy += math.pow(y, 2)
            sum_xy += x * y

        try:
            return sum_xy / math.sqrt(sum_xx * sum_yy)
        except ZeroDivisionError:
            return 0


def main():
    tf_idf = TfIdf()
    top_k = tf_idf.get_top_k_idf(1000, True)
    print(top_k)
    tf_idf.set_bag_of_word(top_k.keys())
    vec1 = tf_idf.get_text_vector(64)
    print(tf_idf.seg_docs[64])
    for i in range(len(tf_idf.seg_docs)):
        vec2 = tf_idf.get_text_vector(i)
        score = tf_idf.cosine_similarity(vec1, vec2)
        if score > 0.5:
            print(tf_idf.seg_docs[i], score)


if __name__ == '__main__':
    main()
