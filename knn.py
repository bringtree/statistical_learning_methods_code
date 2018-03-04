# import tensorflow as tf
# import numpy as np
# import os
# # from tf_knn import KNeighborsClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import numpy as np
#
# pwd = os.getcwd()
#
#
# def onehot(labels):
#   labels = labels.astype(int)
#   onehot_labels = np.eye(labels.shape[0], 31)[labels]
#   return onehot_labels
#
#
# def read_data(x_src, y_src):
#   x = np.load(x_src)
#   y = np.load(y_src)
#
#   return x, y
#
#
# pwd = os.getcwd()
# k_range = list(range(1, 3))
# leaf_range = list(range(1, 2))
# weight_options = ['uniform']
# algorithm_options = ['auto']
# param_gridknn = dict(n_neighbors=k_range, weights=weight_options, algorithm=algorithm_options, leaf_size=leaf_range)
# knn_model = KNeighborsClassifier()
# gridKNN = GridSearchCV(knn_model, param_gridknn, cv=5, n_jobs=-1, scoring='accuracy',verbose=1)
#
# X, Y = read_data(
#   pwd + "/all_X.npy",
#   pwd + "/all_Y.npy")
#
#
# gridKNN.fit(X, Y)
# scores = gridKNN.cv_results_
# accuracy = scores['mean_test_score']
# param_algorithm = scores['param_algorithm']
# param_leaf_size = scores['param_leaf_size']
# param_n_neighbors = scores['param_n_neighbors']
# param_weights = scores['param_weights']
# print("accuracy,n_neighbors,weights,algorithm,leaf_size,,,")
# for i in range(len(accuracy)):
#   print(str(accuracy[i])+','+str(param_n_neighbors[i])+','+str(param_weights[i])+','+str(param_algorithm[i])+','+str(param_leaf_size)+',,,')
# print("end")


import tensorflow as tf
import numpy as np
import os
# from tf_knn import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



pwd = os.getcwd()


def onehot(labels):
  labels = labels.astype(int)
  onehot_labels = np.eye(labels.shape[0], 31)[labels]
  return onehot_labels


def read_data(x_src, y_src):
  x = np.load(x_src)
  y = np.load(y_src)
  return x, y


pwd = os.getcwd()
k_range = list(range(1, 2))
leaf_range = list(range(1, 2))
weight_options = ['uniform']
algorithm_options = ['auto']
param_gridknn = dict(n_neighbors=k_range, weights=weight_options, algorithm=algorithm_options, leaf_size=leaf_range)
knn_model = KNeighborsClassifier()
gridKNN = GridSearchCV(knn_model, param_gridknn, cv=5, n_jobs=-1, scoring='accuracy',verbose=1)

X, Y = read_data(
  pwd + "/0sentence.npy",
  pwd + "/0label.npy")

X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=20, dtype='float32',
                                                      padding='post', truncating='post', value=0.)
nsamples, n,x = X.shape
X = X.reshape((nsamples, n*x))
gridKNN.fit(X, Y)
scores = gridKNN.cv_results_
accuracy = scores['mean_test_score']
param_algorithm = scores['param_algorithm']
param_leaf_size = scores['param_leaf_size']
param_n_neighbors = scores['param_n_neighbors']
param_weights = scores['param_weights']
print("accuracy,n_neighbors,weights,algorithm,leaf_size,,,")
for i in range(len(accuracy)):
  print(str(accuracy[i])+','+str(param_n_neighbors[i])+','+str(param_weights[i])+','+str(param_algorithm[i])+','+str(param_leaf_size)+',,,')
print("end")