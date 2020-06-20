import numpy as np
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import re
import pickle
import string
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

rootpath = Path.cwd()
openfile = open(Path.joinpath(rootpath, r"Wikihow\wiki_data_indep_4"), 'rb')
articles = pickle.load(openfile)
openfile.close()

def create_frame(Dict):
    X = pd.DataFrame(columns=["rel_pos", "Avg-TF-ISF", "rel_len"], dtype=float)
    Y = pd.Series(dtype=int)

    for i in range(len(Dict)):
        indep_vars = Dict["Article{0}".format(i)].iloc[:, [3, 4, 5]]
        dep_vars = Dict["Article{0}".format(i)].in_Summary
        X = X.append(indep_vars, ignore_index=True)
        Y = Y.append(dep_vars, ignore_index=True)
    Y = Y.astype("int")
    
    return X, Y

X, Y = create_frame(articles)
#print(X.head(), Y.head(), X.shape, Y.shape)

for i in X["rel_len"]:
    if type(i) == str:
        print("x" +i+"x")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

BernNB = BernoulliNB(binarize=True)
BernNB.fit(x_train, y_train)
BernNB_y_pred = BernNB.predict(x_test)
BernNB_acc = sklearn.metrics.accuracy_score(y_test, BernNB_y_pred)

MultiNB = MultinomialNB()
MultiNB.fit(x_train, y_train)
MultiNB_y_pred = MultiNB.predict(x_test)
MultiNB_acc = sklearn.metrics.accuracy_score(y_test, MultiNB_y_pred)

GaussiNB = GaussianNB()
GaussiNB.fit(x_train, y_train)
GaussiNB_y_pred = GaussiNB.predict(x_test)
GaussiNB_acc = sklearn.metrics.accuracy_score(y_test, GaussiNB_y_pred)

Logisitc = LogisticRegression()
Logisitc.fit(x_train, y_train)
Logisitc_y_pred = Logisitc.predict(x_test)
Logisitc_acc = sklearn.metrics.accuracy_score(y_test, Logisitc_y_pred)

print("Bern acc: ", BernNB_acc, "\nMulti acc: ", MultiNB_acc, "\nGauss acc: ", GaussiNB_acc, "\nLogisctic acc: ", Logisitc_acc)
