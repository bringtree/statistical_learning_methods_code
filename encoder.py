import jieba
import os
import fastText
import numpy as np
import tensorflow as tf

pwd = "/Users/huangpeisong/Desktop/statistical_learning_methods_code/data_pretreatment/raw_data/raw_data/"

allfiles = os.listdir(pwd)
model = fastText.load_model("/Users/huangpeisong/Downloads/wiki.zh/wiki.zh.bin")
encoder_sentences = []
for file in allfiles:
  with open(pwd + file) as fp:
    data = fp.readlines()
  cut_data = ["/ ".join(jieba.lcut(v, cut_all=False)) for v in data]

  for sentence in data:
    encoder_sentence = []
    jieba_sentence = jieba.lcut(sentence, cut_all=False)
    for word in jieba_sentence:
      encoder_sentence.append(model.get_word_vector(word))

    encoder_sentences.append(encoder_sentence)

  np.save("/Users/huangpeisong/Desktop/statistical_learning_methods_code/numpy_data/" + file.replace("txt", "npy"),
          encoder_sentences)
