import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
openfile = open(Path.joinpath(rootpath, r"Wikihow\wiki_data_indep_8_no_overview_notilte_10k"), 'rb')
articles = pickle.load(openfile)
openfile.close()

def create_frame(Dict, attributes):
    X = pd.DataFrame(columns=attributes, dtype=float)
    Y = pd.Series(dtype=int)

    for i in range(len(Dict)):
        indep_vars = Dict["Article{0}".format(i)][attributes]
        dep_vars = Dict["Article{0}".format(i)].in_Summary
        X = X.append(indep_vars, ignore_index=True)
        Y = Y.append(dep_vars, ignore_index=True)
    Y = Y.astype("int")
    
    return X, Y

#["pos", "rel_pos", "Avg-TF-ISF", "rel_len", "title_sim", "rel_s2s_cohs", "named_ent", "main_con"]
X, Y = create_frame(articles, ["pos", "rel_pos", "Avg-TF-ISF", "rel_len", "rel_s2s_cohs", "centroid_sim", "named_ent", "main_con"])
#print(X.head(), Y.head(), X.shape, Y.shape)


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
"""
BernNB = BernoulliNB(binarize=True)
BernNB.fit(x_train, y_train)
BernNB_y_pred = BernNB.predict(x_test)
BernNB_acc = sklearn.metrics.confusion_matrix(y_test, BernNB_y_pred)

MultiNB = MultinomialNB()
MultiNB.fit(x_train, y_train)
MultiNB_y_pred = MultiNB.predict(x_test)
MultiNB_acc = sklearn.metrics.confusion_matrix(y_test, MultiNB_y_pred)
"""
GaussiNB = GaussianNB() #priors=[0.75, 0.25] makes it worse
GaussiNB.fit(x_train, y_train)
GaussiNB_y_pred = GaussiNB.predict(x_test)
GaussiNB_acc = sklearn.metrics.confusion_matrix(y_test, GaussiNB_y_pred)
GaussiNB_avg_precision_score = sklearn.metrics.average_precision_score(y_test, GaussiNB_y_pred)
"""
Logisitc = LogisticRegression()
Logisitc.fit(x_train, y_train)
Logisitc_y_pred = Logisitc.predict(x_test)
Logisitc_acc = sklearn.metrics.confusion_matrix(y_test, Logisitc_y_pred)
Logisitc_avg_precision_score = sklearn.metrics.average_precision_score(y_test, Logisitc_y_pred)
print("\nLogisctic confusion matrix: \n", Logisitc_acc)
print(, "\nLogisctic avg. precision score: " Logisitc_avg_precision_score)
"""
print("Gauss confusion matrix: \n", GaussiNB_acc)
print("Gauss avg. precision score: ", GaussiNB_avg_precision_score)
#print(sum(GaussiNB_y_pred) /len(GaussiNB_y_pred), sum(Logisitc_y_pred) / len(Logisitc_y_pred))
#print(Y.sum() / Y.shape)

outfile = open(Path.joinpath(rootpath, r"Gauss_trained_10_no_overview_notitle_10k"), 'wb')
articles = pickle.dump(GaussiNB, outfile)
outfile.close()
