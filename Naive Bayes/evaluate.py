import numpy as np
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk import f_measure
from nltk.metrics.scores import recall
from nltk.metrics.scores import precision
from pathlib import Path
import re
import pickle
import string
from rouge import Rouge
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

rootpath = Path.cwd()
openfile = open(Path.joinpath(rootpath, r"Gauss_trained_4_no_overview"), 'rb')
GaussNB = pickle.load(openfile)
openfile.close()

open_wiki = open(Path.joinpath(rootpath, r"Wikihow\wiki_data_indep_4"), 'rb')
wiki_data = pickle.load(open_wiki)
open_wiki.close()

open_cnn_article = open(Path.joinpath(rootpath, r"cnn_indep_4"), 'rb')
cnn_data = pickle.load(open_cnn_article)
open_cnn_article.close()

open_cnn_summ = open(Path.joinpath(rootpath, r"cnn_summaries_dict"), 'rb')
cnn_summary = pickle.load(open_cnn_summ)
open_cnn_summ.close()


def get_summ_cnn(dictionary):
    Summaries = []

    for i in range(len(dictionary)):
        Summary = ""

        for s in range(dictionary["Summary{0}".format(i)].shape[0]):
            Summary += dictionary["Summary{0}".format(i)].iloc[s, 0]
            Summary += " "

        Summary = Summary[:-1]
        Summaries.append(Summary)
    return Summaries
    

def get_summ_wiki(dictionary):
    Summaries = []

    for i in range(len(dictionary)):
        Summary = ""

        for s in range(dictionary["Article{0}".format(i)].shape[0]):
            if dictionary["Article{0}".format(i)].iloc[s, 1] == 1:
                Summary += dictionary["Article{0}".format(i)].iloc[s, 0]
                Summary += " "

        Summary = Summary[:-1]
        Summaries.append(Summary)
    return Summaries


def create_summ_cnn(dictionary):
    Summaries = []

    for i in range(len(dictionary)):
        article = dictionary["Article{0}".format(i)]
        article_props = dictionary["Article{0}".format(i)].drop(["sentence"], axis=1)
        Summary_class = GaussNB.predict(article_props)
        Summary = ""

        for s in range(len(Summary_class)):
            if Summary_class[s] == 1:
                Summary += article.iloc[s, 0]
                Summary += " "
        Summary = Summary[:-1]
        Summaries.append(Summary)

    return Summaries


def create_summ_wiki(dictionary):
    Summaries = []

    for i in range(len(dictionary)):
        article = dictionary["Article{0}".format(i)]
        article_props = dictionary["Article{0}".format(i)].drop(["sentence", "in_Summary"], axis=1)
        Summary_class = GaussNB.predict(article_props)
        Summary = ""

        for s in range(len(Summary_class)):
            if Summary_class[s] == 1:
                Summary += article.iloc[s, 0]
                Summary += " "
        Summary = Summary[:-1]
        Summaries.append(Summary)

    return Summaries


def eval_summ(cnn_summary, cnn_data, wiki_data):
    cnn_summ = get_summ_cnn(cnn_summary)
    cnn_generated_summ = create_summ_cnn(cnn_data)
    wiki_summ = get_summ_wiki(wiki_data)
    wiki_generated_summ = create_summ_wiki(wiki_data)
    rouge = Rouge()
    """
    cnn_scores = np.array([])
    
    for i in range(len(cnn_summ)):
        #score = recall(nltk.sent_tokenize(cnn_summ[i]), nltk.sent_tokenize(cnn_generated_summ[i]))
        print(cnn_generated_summ[i])
        score = rouge.get_scores(cnn_summ[i], cnn_generated_summ[i])
        cnn_scores = np.append(cnn_scores, score)
    cnn_mean_score = cnn_scores.mean()
    print(cnn_mean_score)
    """
    wiki_scores = np.array([])
    for i in range(len(wiki_summ)):
        #score = recall(nltk.sent_tokenize(cnn_summ[i]), nltk.sent_tokenize(cnn_generated_summ[i]))
        print(wiki_summ[i])
        score = rouge.get_scores(wiki_summ[i], wiki_generated_summ[i])
        wiki_scores = np.append(wiki_scores, score)

    print(wiki_scores)    
    wiki_mean_score = wiki_scores.mean()
    


eval_summ(cnn_summary, cnn_data, wiki_data)
