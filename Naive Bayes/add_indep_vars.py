import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import re
import tensorflow_datasets as tfds
import pickle
import string

infile = open("data_dependent_var",'rb')
article_dict = pickle.load(infile)
infile.close()


def pos(Dataframe):
    Dataframe["pos"] = np.linspace(1, Dataframe.shape[0], Dataframe.shape[0])

    return Dataframe


def Relative_pos(Dataframe):
    Dataframe["rel_pos"] = ""
    num_sentences = Dataframe.shape[0]

    for s in range(num_sentences):
        Dataframe["rel_pos"][s] = Dataframe["pos"][s] / num_sentences

    return Dataframe


def TF_ISF(Dataframe):
    sentences = Dataframe["Sentence"]
    sentences = sentences.to_frame()
    ps = nltk.stem.PorterStemmer()
    stop = set(stopwords.words("english"))
    tokenizer = nltk.RegexpTokenizer(r'\w+|\d+')
    num_sentences = sentences.shape[0]

    # Word tokenization, stopword removal, stemming
    for sentence in range(num_sentences): 
        #sentences.iloc[sentence, 0] = sentences.iloc[sentence, 0].translate(string.punctuation)
        #sentences.iloc[sentence, 0] = nltk.word_tokenize(sentences.iloc[sentence, 0])
        sentences.iloc[sentence, 0] = tokenizer.tokenize(sentences.iloc[sentence, 0])
        
        for i in range(len(sentences.iloc[sentence, 0])):
            sentences.iloc[sentence, 0][i] = ps.stem(sentences.iloc[sentence, 0][i].lower())

        sentences.iloc[sentence, 0] = [w for w in sentences.iloc[sentence, 0] if not w in stop]
    #Calculate Term Frequency (within sentence) and Inverse Sentence Frequency
    sentences["TF"] = ""
    sentences["SF"] = ""
    sentences["ISF"] = ""

    for i in range(num_sentences):
        sentences.iloc[i, 1] = []
        sentences.iloc[i, 2] = []
        sentences.iloc[i, 3] = []

        for w in range(len(sentences.iloc[i, 0])):
            TF = 0
            SF = 0

            for v in range(len(sentences.iloc[i, 0])):
                if sentences.iloc[i, 0][w] == sentences.iloc[i, 0][v]:
                    TF += 1
            sentences.iloc[i, 1].append(TF)

            for v in range(num_sentences):
                if sentences.iloc[i, 0][w] in sentences.iloc[v, 0]:
                    SF += 1
            sentences.iloc[i, 2].append(SF)

            ISF = np.log(num_sentences / SF)
            sentences.iloc[i, 3].append(ISF)
    
    # Calculate Averge TF-ISF
    sentences["Avg-TF-ISF"] = ""
    for s in range(num_sentences):
        Avg_TF_ISF = 0

        for w in range(len(sentences.iloc[s, 0])):
            Avg_TF_ISF += sentences.iloc[s, 1][w] * sentences.iloc[s, 3][w] / len(sentences.iloc[s, 0])

        sentences.iloc[s, 4] = Avg_TF_ISF

    sentences.columns = ["Sentence", "TF", "SF", "ISF", "Avg-TF-ISF"]
    #Dataframe["Sentence"] = sentences["Sentence"]
    #Dataframe["TF"] = sentences["TF"]
    #Dataframe["SF"] = sentences["SF"]
    #Dataframe["ISF"] = sentences["ISF"]
    Dataframe["Avg-TF-ISF"] = sentences["Avg-TF-ISF"]

    return Dataframe


def rel_s_lenght(Dataframe): 
    Dataframe["rel_len"] = ""
    num_sentences = Dataframe.shape[0]
    max_words = 0

    for s in range(num_sentences):
        if len(Dataframe.iloc[s, 0]) > max_words:
            max_words = len(Dataframe.iloc[s, 0])
    
    for s in range(num_sentences):
        Dataframe["rel_len"][s] = len(Dataframe.iloc[s, 0]) / max_words

    return Dataframe


def s2s_coherence(Dataframe):
    Dataframe["s2s_coherence"] = ""
    


def add_indep(Dict):
    for i in range(len(Dict)):
        Dict["Article{0}".format(i)] = pos(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = Relative_pos(Dict["Article{0}".format(i)])
        Dict["Article{0}".format(i)] = TF_ISF(Dict["Article{0}".format(i)])
    return(Dict)


def pickle_save(Dict):
    outfile = open("data_indep_3", 'wb')
    pickle.dump(Dict, outfile)
    outfile.close()

test = rel_s_lenght(article_dict["Article0"])
print(test)
"""
add_indep(article_dict)
pickle_save(article_dict)


testopen = open("data_indep_3", 'rb')
idep_dict = pickle.load(testopen)
print(idep_dict["Article200"])
"""