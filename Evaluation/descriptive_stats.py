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
from nltk import f_measure
from nltk.metrics.scores import recall
from nltk.metrics.scores import precision
from pathlib import Path
import re
import pickle
import string
from rouge import Rouge
import matplotlib.pyplot as plt

rootpath = Path.cwd()

## open data for reference summaries:
open_cnn_summ = open(Path.joinpath(rootpath, r"Pre-Processing & EDA\cnn_summaries_dict"), 'rb')
cnn_summary = pickle.load(open_cnn_summ)
open_cnn_summ.close()

open_wiki_summ = open(Path.joinpath(rootpath, r"Wikihow\wiki_partial_summaries_10k"), 'rb')
wiki_summary = pickle.load(open_wiki_summ)
open_wiki_summ.close()

open_wiki = open(Path.joinpath(rootpath, r"Wikihow\wiki_data_indep_8_no_overview_notilte_10k"), 'rb')
wiki_data = pickle.load(open_wiki)
open_wiki.close()

open_cnn_article = open(Path.joinpath(rootpath, r"cnn_indep_8_10k"), 'rb')
cnn_data = pickle.load(open_cnn_article)
open_cnn_article.close()

open_wiki_bayes_summary = open(Path.joinpath(rootpath, r"Naive Bayes\cnn_summary_bayes"), 'rb')
wiki_bayes_summary = pickle.load(open_wiki_bayes_summary)
open_wiki_bayes_summary.close()

open_cnn_bayes_summary = open(Path.joinpath(rootpath, r"Naive Bayes\wiki_summary_bayes"), 'rb')
cnn_bayes_summary = pickle.load(open_cnn_bayes_summary)
open_cnn_bayes_summary.close()


open_wiki_lex_summary = open(Path.joinpath(rootpath, r"LexRank\wiki_TRoutput_summ_dict"), 'rb')
wiki_lex_summary = pickle.load(open_wiki_lex_summary)
open_wiki_lex_summary.close()

open_cnn_lex_summary = open(Path.joinpath(rootpath, r"LexRank\cnn_TRoutput_summ_dict"), 'rb')
cnn_lex_summary = pickle.load(open_cnn_lex_summary)
open_cnn_lex_summary.close()



def get_summ(dictionary):
    length = np.array([])
    for i in range(len(dictionary)):
        length = np.append(length, dictionary["Summary{0}".format(i)].shape[0])

    return length


def get_summ_wiki(dictionary):
    length = np.array([])
    for i in range(len(dictionary)):
        Summary = dictionary["Summary{0}".format(i)]
        Summary = nltk.sent_tokenize(Summary)
        length = np.append(length, len(Summary))

    return length


def get_summ_art(dictionary):
    length = np.array([])
    for i in range(len(dictionary)):
        length = np.append(length, dictionary["Article{0}".format(i)].shape[0])

    return length


def get_bayes_summaries(summaries):
    length = np.array([])
    for i in range(len(summaries)):
        summary = nltk.sent_tokenize(summaries[i])
        length = np.append(length, len(summary))

    return length


def get_hist(ws, cs, wa, ca, wbs, cbs, wls, cls):
    fig, ax = plt.subplots(2, 2)
    fig.suptitle("Lenght of Summary and Aritcle")
    ax[0,0].hist(wa, bins=np.linspace(wa.min(), wa.max(), 30))
    ax[0,0].axvline(wa.mean(), color='k', linestyle='dashed', linewidth=1)
    ax[0,0].set_title("Wikihow Articles")
    ax[0,0].set_xlabel("#Sentences, Mean: 33.84")
    ax[0,1].hist(ca, bins=np.linspace(ca.min(), ca.max(), 30))
    ax[0,1].axvline(ca.mean(), color='k', linestyle='dashed', linewidth=1)
    ax[0,1].set_title("CNN Articles")
    ax[0,1].set_xlabel("#Sentences, Mean: 37.13")
    ax[1,0].boxplot([ws, wbs, wls], labels=["Original", "Bayes", "LexRank"])
    ax[1,0].set_title("#Sentences per Summary (WikiHow)")
    ax[1,1].boxplot([cs, cbs, cls], labels=["Original", "Bayes", "LexRank"])
    ax[1,1].set_title("#Sentences per Summary (CNN)")
    plt.show()


wiki_summ_len = get_summ_wiki(wiki_summary)
cnn_summ_len = get_summ(cnn_summary)
wiki_art_len = get_summ_art(wiki_data)
cnn_art_len = get_summ_art(cnn_data)
wiki_bayes_summ_len = get_bayes_summaries(wiki_bayes_summary)
cnn_bayes_summ_len = get_bayes_summaries(cnn_bayes_summary)
wiki_lex_summ_len = get_summ(wiki_lex_summary)
cnn_lex_summ_len = get_summ(cnn_lex_summary)
cnn_min_sum, cnn_max_sum, cnn_mean_summ = cnn_summ_len.min(), cnn_summ_len.max(), cnn_summ_len.mean()
wiki_min_sum, wiki_max_sum, wiki_mean_summ = wiki_summ_len.min(), wiki_summ_len.max(), wiki_summ_len.mean()
cnn_min, cnn_max, cnn_mean = cnn_art_len.min(), cnn_art_len.max(), cnn_art_len.mean()
wiki_min, wiki_max, wiki_mean = wiki_art_len.min(), wiki_art_len.max(), wiki_art_len.mean()


get_hist(wiki_summ_len, cnn_summ_len, wiki_art_len, cnn_art_len, wiki_bayes_summ_len, cnn_bayes_summ_len, wiki_lex_summ_len, cnn_lex_summ_len)
print(cnn_min_sum, cnn_max_sum, cnn_mean_summ)
print(wiki_min_sum, wiki_max_sum, wiki_mean_summ)

print(cnn_min, cnn_max, cnn_mean)
print(wiki_min, wiki_max, wiki_mean)