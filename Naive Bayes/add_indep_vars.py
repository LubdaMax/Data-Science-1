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

# Add position as indep. variable
for i in range(len(article_dict)):
    article_dict["Article{0}".format(i)]["pos"] = np.linspace(1, article_dict["Article{0}".format(i)].shape[0], article_dict["Article{0}".format(i)].shape[0])   

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
        sentences.iloc[sentence, 0] = [w for w in sentences.iloc[sentence, 0] if not w in stop]

        for i in range(len(sentences.iloc[sentence])):
            sentences.iloc[sentence, 0][i] = ps.stem(sentences.iloc[sentence, 0][i].lower())

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

    sentences[]
    for i in range(num_sentences):


    sentences.columns = ["Sentence", "TF", "SF", "ISF"]
    Dataframe["Sentence"] = sentences["Sentence"]
    Dataframe["TF"] = sentences["TF"]
    Dataframe["SF"] = sentences["SF"]
    Dataframe["ISF"] = sentences["ISF"]
    
    return Dataframe

print(article_dict["Article0"])
test = TF_ISF(article_dict["Article0"])
print(test)

