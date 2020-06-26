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

# This file reads the dictionary with the dataframes for each article with indep. variables.
# Then the Bayesian classifier is trained.
# Only the Wikihow dataset is used for training due to the labeling.
rootpath = Path.cwd()
openfile = open(Path.joinpath(rootpath, r"Wikihow\wiki_data_indep_8_no_overview_notilte_10k"), 'rb')
articles = pickle.load(openfile)
openfile.close()

def create_frame(Dict, attributes):
    """creates the dataframe usable for scikit learn methods.
    For stepwise feature selection purposes the attributes used
    for training have to be specified."""
    X = pd.DataFrame(columns=attributes, dtype=float) # Dataframe with indep. variables
    Y = pd.Series(dtype=int) # Series with dep. variables

    for i in range(len(Dict)): # Substitute range(len(Dict) - 500) 
        # This leaves the last 500 Articles as validation set.
        indep_vars = Dict["Article{0}".format(i)][attributes]
        dep_vars = Dict["Article{0}".format(i)].in_Summary
        X = X.append(indep_vars, ignore_index=True)
        Y = Y.append(dep_vars, ignore_index=True)
    Y = Y.astype("int")
    
    return X, Y

def stepwsie_select(Dict):
    """Executes forward stepwise selection of features. The features are
    tested on a separate set from their training data.
    All features were used in final model."""
    GaussiNB = GaussianNB() # The classifier
    attributes = ["pos", "rel_pos", "Avg-TF-ISF", "rel_len", "rel_s2s_cohs", "centroid_sim", "named_ent", "main_con"]
    precision = [] # Holds precision values of selected features
    selected = [] # Holds selected features
    for i in range(len(attributes)):
        current = np.array([]) # holds the currently tested feature       
        for j in range(len(attributes)):
            # Based on selected features, classifier is trained and tested
            selected.append(attributes[j])

            X, Y = create_frame(Dict, selected)
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

            GaussiNB.fit(x_train, y_train)
            GaussiNB_y_pred = GaussiNB.predict(x_test)
            GaussiNB_avg_precision_score = sklearn.metrics.average_precision_score(y_test, GaussiNB_y_pred) 

            current = np.append(current, GaussiNB_avg_precision_score)
            selected = selected[:-1]
        
        # select the best performing feature and add it to selected
        best = current.argmax()
        precision.append(current.max())
        selected.append(attributes[best])
        del attributes[best]
        print(selected)
        print(precision)

# Execute forward stepwise selection
# stepwsie_select(articles)

# Creating the frame for all features
X, Y = create_frame(articles, ["pos", "rel_pos", "Avg-TF-ISF", "rel_len", "rel_s2s_cohs", "centroid_sim", "named_ent", "main_con"])
#print(X.head(), Y.head(), X.shape, Y.shape) # for testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

# Training and running preliminary accuracy measures.
# !!The Classifier used for the Project was trained using only part of the
# Articles, leaving the last 500 for validation rouge scores!!
GaussiNB = GaussianNB() # custom priors=[0.75, 0.25] decreases performance
GaussiNB.fit(x_train, y_train)
GaussiNB_y_pred = GaussiNB.predict(x_test)
GaussiNB_acc = sklearn.metrics.confusion_matrix(y_test, GaussiNB_y_pred)
GaussiNB_avg_precision_score = sklearn.metrics.average_precision_score(y_test, GaussiNB_y_pred)

print(GaussiNB_acc)
print(GaussiNB_avg_precision_score)
outfile = open(Path.joinpath(rootpath, r"Naive Bayes\Gauss_trained_10_no_overview_notitle_10k"), 'wb')
articles = pickle.dump(GaussiNB, outfile)
outfile.close()


# Eval---------------------------------------------------------------------------------------------------

rootpath = Path.cwd()

## open data for reference summaries:
open_cnn_summ = open(Path.joinpath(rootpath, r"Pre-Processing & EDA\cnn_summaries_dict"), 'rb')
cnn_summary = pickle.load(open_cnn_summ)
open_cnn_summ.close()
open_wiki_summ = open(Path.joinpath(rootpath, r"Wikihow\wiki_partial_summaries_10k"), 'rb')
wiki_summary = pickle.load(open_wiki_summ)
open_wiki_summ.close()


## open data for output summaries generated by NAIVE BAYES
openfile = open(Path.joinpath(rootpath, r"Naive Bayes\Gauss_trained_10_no_overview_notitle_10k"), 'rb')
GaussNB = pickle.load(openfile)
openfile.close()

open_wiki = open(Path.joinpath(rootpath, r"Wikihow\wiki_data_indep_8_no_overview_notilte_10k"), 'rb')
wiki_data = pickle.load(open_wiki)
open_wiki.close()


open_cnn_article = open(Path.joinpath(rootpath, r"Naive Bayes\cnn_indep_8_10k"), 'rb')
cnn_data = pickle.load(open_cnn_article)
open_cnn_article.close()



def get_summ(dictionary):
    """Gets summaries out of dictionaries and returens list
    of summaries for easy analysis"""
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
    """Gets summaries out of dictionaries and returens list
    of summaries for easy analysis"""
    Summaries = []
    # Only last 500 summaries are evaluated (test set)
    for i in range(len(dictionary)-500, len(dictionary)):
        Summary = dictionary["Summary{0}".format(i)]
        Summaries.append(Summary)
    return Summaries


def create_summ_NB_cnn(dictionary):
    """Applies the classifier to the cnn data and returns list of summaries.
    Input is a dictionary of Dataframes containing the articles, output
    is a list of generated summaries.
    Classifier is imported from file created above."""
    Summaries = []
    rootpath = Path.cwd()
    openfile = open(Path.joinpath(rootpath, r"Naive Bayes\Gauss_trained_10_no_overview_notitle_10k"), 'rb')
    GaussNB = pickle.load(openfile)
    openfile.close()
    for i in range(len(dictionary)):
        # The first column containing the raw string is dropped.
        # The clasifier works only on the features.
        article = dictionary["Article{0}".format(i)]
        article_props = dictionary["Article{0}".format(i)].drop(["sentence"], axis=1)
        Summary_class = GaussNB.predict(article_props)
        Summary = ""

        # format summaries properly
        for s in range(len(Summary_class)):
            if Summary_class[s] == 1:
                Summary += article.iloc[s, 0]
                Summary += " "
        Summary = Summary[:-1]
        Summaries.append(Summary)
    
    # Saves the summaries to pikle file
    outfile = open(Path.joinpath(rootpath, r"Naive Bayes\cnn_summary_bayes"), 'wb')
    pickle.dump(Summaries, outfile)
    outfile.close()
    return Summaries


def create_summ_NB_wiki(dictionary):
    """Applies the classifier to the Wikihow data and returns list of summaries.
    Input is a dictionary of Dataframes containing the articles, output
    is a list of generated summaries.
    Classifier is imported from file created above."""
    Summaries = []
    rootpath = Path.cwd()
    openfile = open(Path.joinpath(rootpath, r"Naive Bayes\Gauss_trained_10_no_overview_notitle_10k"), 'rb')
    GaussNB = pickle.load(openfile)
    openfile.close()
    for i in range(len(dictionary)-500, len(dictionary)):
        # Only last 500 summaries are evaluated (test set)
        # The first two columns containing the raw string and dep. var. are dropped.
        # The clasifier works only on the features.
        article = dictionary["Article{0}".format(i)]
        article_props = dictionary["Article{0}".format(i)].drop(["sentence", "in_Summary"], axis=1)
        Summary_class = GaussNB.predict(article_props)
        Summary = ""

        # format summaries properly
        for s in range(len(Summary_class)):
            if Summary_class[s] == 1:
                Summary += article.iloc[s, 0]
                Summary += " "
        Summary = Summary[:-1]
        Summaries.append(Summary)

    # Saves the summaries to pikle file
    outfile = open(Path.joinpath(rootpath, r"Naive Bayes\wiki_summary_bayes"), 'wb')
    pickle.dump(Summaries, outfile)
    outfile.close()
    return Summaries


def eval_summ(cnn_summary, cnn_data, wiki_summary, wiki_data):
    """Evaluates the quality of the summaies of the bayesian classifier using rouge scores."""
    cnn_summ = get_summ(cnn_summary) # creates list of provided summaries
    cnn_generated_summ = create_summ_NB_cnn(cnn_data) # creates list of generated summaries
    wiki_summ = get_summ_wiki(wiki_summary) # creates list of provided summaries
    wiki_generated_summ = create_summ_NB_wiki(wiki_data) # creates list of generated summaries
    rouge = Rouge() # The libariy containing the rouge score

    # The various scores are generated and saved in a dataframe (for the cnn data)
    cnn_score_frame = pd.DataFrame(columns=["r1-f", "r1-p", "r1-r", "r2-f", "r2-p", "r2-r", "rl-f", "rl-p", "rl-r"])
    for i in range(len(cnn_summ)):
        if (len(cnn_generated_summ[i]) and len(cnn_summ[i])) > 0:
            try:
                score = rouge.get_scores(cnn_generated_summ[i], cnn_summ[i])

                cnn_score_frame = cnn_score_frame.append(pd.DataFrame(
                    {'r1-f': score[0]['rouge-1']['f'], 'r1-p': score[0]['rouge-1']['p'],
                     'r1-r': score[0]['rouge-1']['r'], 'r2-f': score[0]['rouge-2']['f'],
                     'r2-p': score[0]['rouge-2']['p'], 'r2-r': score[0]['rouge-2']['r'],
                     'rl-f': score[0]['rouge-l']['f'], 'rl-p': score[0]['rouge-l']['p'],
                     'rl-r': score[0]['rouge-l']['r']}, index=[0]), ignore_index=True)
            except:
                # The exception handles faulty summaries
                # This exception triggers only once on our dataset because one generated
                # summary contains only a dot which was most likely tokenized incorrectly.
                print("summary: ", cnn_summ[i])
                print("generated: ", cnn_generated_summ[i])
                print(
                    "------------------------------------------------------------------------------------------------------------------------")

    # The various scores are generated and saved in a dataframe (for the wikihow data)
    wiki_score_frame = pd.DataFrame(columns=["r1-f", "r1-p", "r1-r", "r2-f", "r2-p", "r2-r", "rl-f", "rl-p", "rl-r"])
    for i in range(len(wiki_summ)):
        if (len(wiki_generated_summ[i]) and len(wiki_summ[i])) > 0:
            score = rouge.get_scores(wiki_generated_summ[i], wiki_summ[i])

            wiki_score_frame = wiki_score_frame.append(pd.DataFrame(
                {'r1-f': score[0]['rouge-1']['f'], 'r1-p': score[0]['rouge-1']['p'], 'r1-r': score[0]['rouge-1']['r'],
                 'r2-f': score[0]['rouge-2']['f'], 'r2-p': score[0]['rouge-2']['p'], 'r2-r': score[0]['rouge-2']['r'],
                 'rl-f': score[0]['rouge-l']['f'], 'rl-p': score[0]['rouge-l']['p'], 'rl-r': score[0]['rouge-l']['r']},
                index=[0]), ignore_index=True)

    return wiki_score_frame, cnn_score_frame

##evaluate performance NAIVE BAYES and print mean measurements
# We report only the r1/r2 p/r scores for interpretability
wiki_score, cnn_score = eval_summ(cnn_summary, cnn_data, wiki_summary, wiki_data)
print(wiki_score.head(), wiki_score.shape)
print("r1-f mean: ", wiki_score["r1-f"].mean())
print("r1-p mean: ", wiki_score["r1-p"].mean())
print("r1-r mean: ", wiki_score["r1-r"].mean())

print("r2-f mean: ", wiki_score["r2-f"].mean())
print("r2-p mean: ", wiki_score["r2-p"].mean())
print("r2-r mean: ", wiki_score["r2-r"].mean())

print("rl-f mean: ", wiki_score["rl-f"].mean())
print("rl-p mean: ", wiki_score["rl-p"].mean())
print("rl-r mean: ", wiki_score["rl-r"].mean())
print(cnn_score.head(), cnn_score.shape)
print("r1-f mean: ", cnn_score["r1-f"].mean())
print("r1-p mean: ", cnn_score["r1-p"].mean())
print("r1-r mean: ", cnn_score["r1-r"].mean())

print("r2-f mean: ", cnn_score["r2-f"].mean())
print("r2-p mean: ", cnn_score["r2-p"].mean())
print("r2-r mean: ", cnn_score["r2-r"].mean())

print("rl-f mean: ", cnn_score["rl-f"].mean())
print("rl-p mean: ", cnn_score["rl-p"].mean())
print("rl-r mean: ", cnn_score["rl-r"].mean())
