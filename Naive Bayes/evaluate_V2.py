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


os.chdir("C:/Users/Leni/Google Drive/00_Studium/01_Master WI Goethe/01_Veranstaltungen/SS20_DS_Data Science 1/DS Project/NLP _Text Summarizer/")
rootpath = Path.cwd()

## open Data for Reference Summaries

openfile = open(Path.joinpath(rootpath, r"LexRank\cnn_summaries_dict"), 'rb')
cnn_summary = pickle.load(openfile)
openfile.close()

# openfile = open(Path.joinpath(rootpath, r"Wikihow\wiki_partial_summaries"), 'rb')
# wiki_summary = pickle.load(openfile)
# openfile.close()

## Open Output Summaries: NAIVE BAISE

# openfile = open(Path.joinpath(rootpath, r"Gauss_trained_8_no_overview"), 'rb')
# GaussNB = pickle.load(openfile)
# openfile.close()
#
# openfile = open(Path.joinpath(rootpath, r"Wikihow\wiki_data_indep_8_no_overview"), 'rb')
# wiki_data = pickle.load(openfile)
# openfile.close()
#
# openfile = open(Path.joinpath(rootpath, r"cnn_indep_8"), 'rb')
# cnn_data = pickle.load(openfile)
# openfile.close()

## Open Output Summaries: TEXT RANK

openfile = open(Path.joinpath(rootpath, r"LexRank\cnnTRoutput_summ_dict"), 'rb')
cnn_data = pickle.load(openfile)
openfile.close()

# openfile = open(Path.joinpath(rootpath, r"XXX"), 'rb')
# wiki_data = pickle.load(openfile)
# openfile.close()


filename = r"LexRank\wikiTRoutput_summ_dict"
outfile = open(Path.joinpath(rootpath, filename), 'wb')
pickle.dump(TRoutput_summ_dict, outfile)
outfile.close()

filename = r"LexRank\wiki_TRoutput_ranking_dict"
outfile = open(Path.joinpath(rootpath, filename), 'wb')
pickle.dump(ranking_dict, outfile)
outfile.close()



def TR_generate_summ(ranked_sentences_dict, article_key, article_data, number_sentences):
    """ selects specified number of ranked sentences"
    :param ranked_sentences_dict: dictionary that ranks sentences this way:
    :param article_key:
    :param article_data:
    :param number_sentences: number of sentences in summary
    :return:
    """
    summary = []
    for i in range(number_sentences):
        s = ranked_sentences_dict[article_key][i][0]
        summary.append(article_data[article_key].iloc[s, 0])
        summary.append(" ")

        #summary = "".join(summary)

    return summary



def get_summ(dictionary):
    """
    :param dictionary:
    :return:
    """
    Summaries = []

    for i in range(len(dictionary)):
        Summary = ""

        for s in range(dictionary["Summary{0}".format(i)].shape[0]):
            Summary += dictionary["Summary{0}".format(i)].iloc[s, 0]
            Summary += " "

        Summary = Summary[:-1]
        Summaries.append(Summary)

    return Summaries




def eval_Rouge (reference_Summary, output_Summary):
    """

    :param reference_Summary: list as input
    :param output_Summary: list as input
    :return:
    """
    scores = np.array([])
    for i in range(len(reference_Summary)):
        # score = recall(nltk.sent_tokenize(reference_Summary[i]), nltk.sent_tokenize(output_Summary[i]))
        print(wiki_summ[i])
        score = rouge.get_scores(reference_Summary[i], output_Summary[i])
        scores = np.append(scores, score)

    print(scores)
    wiki_mean_score = wiki_scores.mean()
    return score



open_cnn_article = open(Path.joinpath(rootpath, r"cnn_indep_8"), 'rb')
cnn_data = pickle.load(open_cnn_article)
open_cnn_article.close()

print(cnn_data)