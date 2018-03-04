import numpy as np
import os

train_sentences = np.array([])
train_labels = np.array([])

test_sentences = np.array([])
test_labels = np.array([])


def k_fold(row_data_src, labels_index, k=5):
  row_data = np.load(row_data_src)
  np.random.shuffle(row_data)
  length = len(row_data)
  split_point = int(length / k)
  tmp_test_sentences = row_data[0:split_point]
  tmp_test_labels = np.ones(split_point) * labels_index
  tmp_train_sentences = row_data[split_point:length]
  tmp_train_labels = np.ones(length - split_point) * labels_index
  return tmp_test_sentences, tmp_test_labels, tmp_train_sentences, tmp_train_labels


parents_src = "/Users/huangpeisong/Desktop/statistical_learning_methods_code/numpy_data/"
list_files = os.listdir(parents_src)
labels_index = 0
for file in list_files:
  labels_index += 1
  tmp_test_sentences, tmp_test_labels, tmp_train_sentences, tmp_train_labels = k_fold(parents_src + file, labels_index,1)
  train_sentences = np.append(train_sentences, tmp_train_sentences)
  train_labels = np.append(train_labels, tmp_train_labels)
  test_sentences = np.append(test_sentences, tmp_test_sentences)
  test_labels = np.append(test_labels, tmp_test_labels)

index = 0
np.save(str(index) + "sentence.npy", test_sentences)
np.save(str(index) + "label.npy", test_labels)
