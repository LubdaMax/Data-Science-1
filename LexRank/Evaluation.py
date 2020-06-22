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


os.chdir("C:/Users/Leni/Google Drive/00_Studium/01_Master WI Goethe/01_Veranstaltungen/SS20_DS_Data Science 1/DS Project/NLP _Text Summarizer/")
rootpath = Path.cwd()

openfile = open(Path.joinpath(rootpath, r"cnnTRoutput_summ_dict"), 'rb')
cnn_TRoutput_summ_dict = pickle.load(openfile)
openfile.close()

openfile = open(Path.joinpath(rootpath, r"cnn_summaries_dict"), 'rb')
cnn_reference_summ_dict = pickle.load(openfile)
openfile.close()


openfile = open(Path.joinpath(rootpath, r"cnn_TRoutput_ranking_dict"), 'rb')
cnn_TRoutput_ranking_dict = pickle.load(openfile)
openfile.close()

openfile = open(Path.joinpath(rootpath, r"cnn_dataframe, 'rb')
cnn_data = pickle.load(openfile)
openfile.close()



def generate_output_summ(ranked_sentences_dict, article_key, article_data, number_sentences):
    summary = []
    for i in range(number_sentences):
        s = ranked_sentences_dict[article_key][i][0]
        summary.append(article_data[article_key].iloc[s, 0])
        summary.append(" ")

        summary = "".join(summary)

    return summary



def get_summ(dictionary):
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
    scores = np.array([])
    for i in range(len(reference_Summary)):
        # score = recall(nltk.sent_tokenize(reference_Summary[i]), nltk.sent_tokenize(output_Summary[i]))
        print(wiki_summ[i])
        score = rouge.get_scores(reference_Summary[i], output_Summary[i])
        scores = np.append(scores, score)

    print(scores)
    wiki_mean_score = wiki_scores.mean()
    return score




# create input (reference summary and generated output summaries as lists)
cnnTRoutput_summ = get_summ(cnnTRoutput_summ_dict)
cnn_reference_summ = get_summ(cnn_reference_summ_dict)


# evaluate performance with ROUGE (output generated in Summarizer.py)
rouge = Rouge()
evalR = eval_Rouge(cnn_reference_summ, cnnTRoutput_summ)


# evaluate performance with ROUGE (generating a variance of outputs with different number of sentences)
evalR_var_results = []
for i in range(10):
    evalR_var = []
    for key in cnn_reference_summ_dict.keys():
        output_var = generate_output_summ(cnn_TRoutput_ranking_dict, key, cnn_data, i)
        evalR_var.append(i, eval_Rouge(cnn_reference_summ, output_var))

    evalR_var_results.append(i, evalR_var.mean())
    print("average score for ", i, "-sentence Summary: ", evalR_var.mean())






