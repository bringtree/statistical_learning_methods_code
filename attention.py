import tensorflow as tf
import jieba
import fastText
import numpy as np

PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'

epoch = 50
epoch_period = 10

max_seq_len = 20

encoder_num_layers = 1
decoder_num_layers = 2

encoder_drop_prob = 0.1
decoder_drop_prob = 0.1

encoder_hidden_dim = 256
decoder_hidden_dim = 256
alignment_dim = 25

input_seqs = []
output_seqs = []


def read_data(input_len):
  with open("./data_pretreatment/all_train_raw_data.txt") as fp:
    lines = fp.readlines()
    for line in lines:
      input_seq, output_seq = line.rsplit('\t')
      tmp = []
      tmp.append(BOS)
      input_seq = jieba.lcut(input_seq, cut_all=False)
      tmp += input_seq
      tmp.append(EOS)
      if '\ufeff' in tmp:
        tmp.remove('\ufeff')
      while len(tmp) < input_len:
        tmp.append(PAD)

      input_seqs.append(tmp)

      output_seqs.append(output_seq)

      # fr_vocab = text.vocab.Vocabulary(collections.Counter(input_tokens),
      #                                    reserved_tokens=[PAD, BOS, EOS])
      # en_vocab = text.vocab.Vocabulary(collections.Counter(output_tokens),
      #                                    reserved_tokens=[PAD, BOS, EOS])
      # return fr_vocab, en_vocab, input_seqs, output_seqs


read_data(20)
print('1')
