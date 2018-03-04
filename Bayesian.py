import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import matplotlib.pyplot as plt

pwd = os.getcwd()

n_pwd = "/data_pretreatment/data_other/"


def gaussianNB():
  gnb = GaussianNB()
  score = 0
  for i in range(5):
    test_x = np.load(pwd + n_pwd + str(i) + "_train_test_x_test_kf_smp_bow.npy")
    train_x = np.load(pwd + n_pwd + str(i) + "_train_test_x_train_kf_smp_bow.npy")
    test_y = np.load(pwd + n_pwd + str(i) + "_train_test_y_test_kf_smp_bow.npy")
    train_y = np.load(pwd + n_pwd + str(i) + "_train_test_y_train_kf_smp_bow.npy")
    score += gnb.fit(train_x, train_y).score(test_x, test_y)
  print(score / 5)


def multinomialNB():
  alphas = np.logspace(-2, 0, num=200)
  testing_score = []
  for alpha in alphas:
    gnb = MultinomialNB(alpha)
    score = 0
    for i in range(5):
      test_x = np.load(pwd + n_pwd + str(i) + "_train_test_x_test_kf_smp_bow.npy")
      train_x = np.load(pwd + n_pwd + str(i) + "_train_test_x_train_kf_smp_bow.npy")
      test_y = np.load(pwd + n_pwd + str(i) + "_train_test_y_test_kf_smp_bow.npy")
      train_y = np.load(pwd + n_pwd + str(i) + "_train_test_y_train_kf_smp_bow.npy")
      score += gnb.fit(train_x, train_y).score(test_x, test_y)
    testing_score.append(score / 5)
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(alphas, testing_score, label="testing score")
  ax.set_xlabel('alpha')
  ax.set_ylabel('score')
  ax.set_title("MultinomoalNB")
  ax.set_xscale("log")
  plt.show()

def bernoulliNB():
  alphas = np.logspace(-2, 5, num=200)
  testing_score = []
  for alpha in alphas:
    gnb = BernoulliNB(alpha=alpha)
    score = 0
    for i in range(5):
      test_x = np.load(pwd + n_pwd + str(i) + "_train_test_x_test_kf_smp_bow.npy")
      train_x = np.load(pwd + n_pwd + str(i) + "_train_test_x_train_kf_smp_bow.npy")
      test_y = np.load(pwd + n_pwd + str(i) + "_train_test_y_test_kf_smp_bow.npy")
      train_y = np.load(pwd + n_pwd + str(i) + "_train_test_y_train_kf_smp_bow.npy")
      score += gnb.fit(train_x, train_y).score(test_x, test_y)
    testing_score.append(score / 5)
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(alphas, testing_score, label="testing score")
  ax.set_xlabel('alpha')
  ax.set_ylabel('score')
  ax.set_title("MultinomoalNB")
  ax.set_xscale("log")
  plt.show()