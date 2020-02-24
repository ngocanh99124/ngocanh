import os
import re
import cv2
import sys
import time
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from tools import *
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression

def func(x):
    return -x[0]

def load_weakClassifier(number):
    path_model = "weak_model"
    tuple_ = []
    for i in range(number):
        path = path_model+str(i)+".pickle"
        (clf, alpha_m) = pickle.load(open(path, "rb"))
        tuple_.append((clf, alpha_m))
    return tuple_

def predict_adaboost(feat, list_clf):
    x = 0
    for clf, alpha_m in list_clf:
        x = x + clf.predict(feat)[0]*alpha_m
    return np.sign(x), x

def extract(filename, list_clf, keep=3):
    lst = []
    images = cv2.imread(filename, 0)
    print(images.shape)
    for i, image in enumerate(pyramid(images, min_size=64, step=.75)):
        w_ = (64 * images.shape[0]) // image.shape[0]
        if w_ < np.min([images.shape[0], images.shape[1]])//2:
            continue
        for (x, y, img) in sliding_window(image, 32*image.shape[0]//images.shape[0], (64, 64)):

            if img.shape != (64, 64):
                continue
            x_ = (y * images.shape[0]) // image.shape[0]
            y_ = (x * images.shape[0]) // image.shape[0]

            #print(x_, y_, w_)
            feat = pre_processing(img)
            #cv2.imshow("aa", img)
            #cv2.imshow("bb", image[x_:x_+w_,y_:y_+w_])
            #cv2.waitKey(0)
            sign, weight = predict_adaboost([feat], list_clf)
            if sign == 1:
                #lst.append(img)
                lst.append((weight, x_, y_, w_))
    if len(lst) == 0:
        return []
    lst = sorted(lst, key = func)
    lst = lst[:keep]
    print(lst)
    return lst
    #return lst
number = 200
list_clf = load_weakClassifier(number)

def render_candidates(image, candidates):
    canvas = image
    print(canvas.shape)
    for (x, y, xx, yy) in candidates:
        canvas[x:xx, y, :] = [1., 0., 0.]
        canvas[x:xx, yy, :] = [1., 1., 0.]
        canvas[x, y:yy, :] = [0., 1., 0.]
        canvas[xx, y:yy, :] = [0., 0., 1.]
    return image


def bounding_box(path):
    img = cv2.imread(path, 1)
    lst = extract(path, list_clf)
    new_candi = []
    for (val, x, y, w) in lst:
        new_candi.append((x, y, x+w, y+w))
    img = render_candidates(img, new_candi)
    cv2.imshow("ab",img)
    cv2.waitKey(0)


path = "C:/Users/ThinkPad/Desktop/12.jpg"
bounding_box(path)
