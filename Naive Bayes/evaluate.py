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
openfile = open(Path.joinpath(rootpath, r"Gauss_trained_8_no_overview"), 'rb')
GaussNB = pickle.load(openfile)
openfile.close()

open_wiki = open(Path.joinpath(rootpath, r"Wikihow\wiki_data_indep_8_no_overview"), 'rb')
wiki_data = pickle.load(open_wiki)
open_wiki.close()

open_wiki_summ = open(Path.joinpath(rootpath, r"Wikihow\wiki_partial_summaries"), 'rb')
wiki_summary = pickle.load(open_wiki_summ)
open_wiki_summ.close()

open_cnn_article = open(Path.joinpath(rootpath, r"cnn_indep_8"), 'rb')
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
        Summary = dictionary["Summary{0}".format(i)]
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


def eval_summ_bayes(cnn_summary, cnn_data, wiki_data, wiki_summary):
    cnn_summ = get_summ_cnn(cnn_summary)
    cnn_generated_summ = create_summ_cnn(cnn_data)
    wiki_summ = get_summ_wiki(wiki_summary)
    wiki_generated_summ = create_summ_wiki(wiki_data)
    rouge = Rouge()
    
    cnn_score_frame = pd.DataFrame(columns=["r1-f", "r1-p", "r1-r", "r2-f", "r2-p", "r2-r","rl-f", "rl-p", "rl-r"])
    for i in range(len(cnn_summ)):
        #score = recall(nltk.sent_tokenize(cnn_summ[i]), nltk.sent_tokenize(cnn_generated_summ[i]))
        if (len(cnn_summ[i]) and len(cnn_generated_summ[i])) > 0: 
            try: 
                score = rouge.get_scores(cnn_summ[i], cnn_generated_summ[i])

                cnn_score_frame = cnn_score_frame.append(pd.DataFrame({'r1-f': score[0]['rouge-1']['f'], 'r1-p': score[0]['rouge-1']['p'], 'r1-r': score[0]['rouge-1']['r'], 'r2-f': score[0]['rouge-2']['f'], 'r2-p': score[0]['rouge-2']['p'], 'r2-r': score[0]['rouge-2']['r'], 'rl-f': score[0]['rouge-l']['f'], 'rl-p': score[0]['rouge-l']['p'], 'rl-r': score[0]['rouge-l']['r']}, index=[0]), ignore_index=True)
            except:
                print("summary: ", cnn_summ[i])
                print("generated: ", cnn_generated_summ[i])
                print("------------------------------------------------------------------------------------------------------------------------")

    wiki_score_frame = pd.DataFrame(columns=["r1-f", "r1-p", "r1-r", "r2-f", "r2-p", "r2-r","rl-f", "rl-p", "rl-r"])
    for i in range(len(wiki_summ)):
        #score = recall(nltk.sent_tokenize(cnn_summ[i]), nltk.sent_tokenize(cnn_generated_summ[i]))
        if (len(wiki_summ[i]) and len(wiki_generated_summ[i])) > 0: 
            score = rouge.get_scores(wiki_summ[i], wiki_generated_summ[i])

            wiki_score_frame = wiki_score_frame.append(pd.DataFrame({'r1-f': score[0]['rouge-1']['f'], 'r1-p': score[0]['rouge-1']['p'], 'r1-r': score[0]['rouge-1']['r'], 'r2-f': score[0]['rouge-2']['f'], 'r2-p': score[0]['rouge-2']['p'], 'r2-r': score[0]['rouge-2']['r'], 'rl-f': score[0]['rouge-l']['f'], 'rl-p': score[0]['rouge-l']['p'], 'rl-r': score[0]['rouge-l']['r']}, index=[0]), ignore_index=True)

    return wiki_score_frame, cnn_score_frame
    
    
wiki_score, cnn_score = eval_summ_bayes(cnn_summary, cnn_data, wiki_data, wiki_summary)
print(wiki_score.head(), wiki_score.shape)
print("r1-f mean: ", wiki_score["r1-f"].mean())
print("r2-f mean: ", wiki_score["r2-f"].mean())
print("rl-f mean: ", wiki_score["rl-f"].mean())
print(cnn_score.head(), cnn_score.shape)
print("r1-f mean: ", cnn_score["r1-f"].mean())
print("r2-f mean: ", cnn_score["r2-f"].mean())
print("rl-f mean: ", cnn_score["rl-f"].mean())