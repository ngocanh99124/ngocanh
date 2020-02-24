from __future__ import division, print_function
import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_hastie_10_2


# Import helper functions

import os
import re
import cv2
import time
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tools import *
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression

NEGATIVES = "dataset/negatives"
POSITIVES = "dataset/container"



    #loading dataset
positives = []
negatives = []
positives_names = os.listdir(POSITIVES)
negatives_names = os.listdir(NEGATIVES)


random.shuffle(positives_names)
random.shuffle(negatives_names)

#positives_names = positives_names[:min(len(positives_names), len(negatives_names))]
#negatives_names = negatives_names[:min(len(positives_names), len(negatives_names))]
print(len(positives_names), len(negatives_names))
print ("load positives...")


scores = []
for img_name in tqdm(positives_names):
    try:
        positives.append((cv2.resize(imread(POSITIVES + '/'+img_name), (64, 64)), 1))
    except: pass

print ("load negatives...")

random.shuffle(negatives_names)
negatives_names = negatives_names[:len(positives_names)]
for img_name in tqdm(negatives_names):
    try:
        negatives.append((cv2.resize(imread(NEGATIVES + '/'+img_name),(64, 64)), -1))
    except: pass

dataset = positives + negatives
dataset = random.sample(dataset, len(dataset))
new_dataset = []
for img, val in dataset:
    new_dataset.append((pre_processing(img), val))
dataset = new_dataset

print(len(dataset))

"""HELPER FUNCTION: GET ERROR RATE == == == == == == == == == == == == == == == == == == == == = """
def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

"""HELPER FUNCTION: PRINT ERROR RATE == == == == == == == == == == == == == == == == == == == = """
def print_error_rate(err):
    print('Error rate: Training: %.4f - Test: %.4f' % err)

"""HELPER FUNCTION: GENERICCLASSIFIER == == == == == == == == == == == == == == == == == == = """
def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)

"""ADABOOSTIMPLEMENTATION == == == == == == == == == == == == == == == == == == == == == == == == = """
def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(M):
        # Fit a classifier with the specific weights
        print("train weak model %s" %str(i))
        clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        # Error
        err_m = np.dot(w,miss) / sum(w)

        # Alpha
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        print(err_m, alpha_m)
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, 
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test,
                                         [x * alpha_m for x in pred_test_i])]
        tuple_ = (clf, alpha_m)
        pickle.dump(tuple_, open("weak_model"+str(i)+".pickle", "wb"))
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)

"""PLOTFUNCTION == == == == == == == == == == == == == == == == == == == == == == == == == == == == == = """
def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6),
            color = ['lightblue', 'darkblue'], grid = True)
    plot1.set_xlabel('Number of iterations', fontsize = 12)
    plot1.set_xticklabels(range(0,450,50))
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title('Error rate vs number of iterations', fontsize = 16)
    plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')


train_set, test_set = random_split(dataset, .8)
X_train1, y_train = zip(*train_set)
X_test1, y_test = zip(*test_set)
X_train = []
for arr in X_train1:
    X_train.append(arr.tolist())
X_test = []
for arr in X_test1:
    X_test.append(arr.tolist())
X_train = np.array(X_train)
X_test = np.array(X_test)
clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
number_stage = 10
er_tree = generic_clf(y_train, X_train, y_test, X_test, clf_tree)

# Fit Adaboost classifier using a decision tree as base estimator
# Test with different number of iterations
er_train, er_test = [er_tree[0]], [er_tree[1]]


er_i = adaboost_clf(y_train, X_train, y_test, X_test, 200, clf_tree)
er_train.append(er_i[0])
er_test.append(er_i[1])
print(er_i)

# Compare error rate vs number of iterations
plot_error_rate(er_train, er_test)

