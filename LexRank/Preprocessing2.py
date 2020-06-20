import numpy as np
import pandas as pd
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfTransformer
import networkx as nx
import collections
from collections import Counter
import tensorflow as tf

# Initialize the dataframe used to store Articles in col. 0         ######## TO DO:  fit Array size to article categories
index = np.linspace(0, 510, 511)
total_data = pd.DataFrame(columns=["Category","Summary","Article_input","Article_prep","Article_prep2","Summary_output"], index=index, dtype = str)
total_data = total_data.fillna(0)

#
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)

# Relative path of dataset
rootpath = Path.cwd()
articlepath_folder = Path.joinpath(rootpath, r"Dataset 1 (BBC)\News Articles")
summarypath_folder = Path.joinpath(rootpath, r"Dataset 1 (BBC)\Summaries")


# Insert data from .txt files into the dataframe
categories = ["business"]
#categories = ["business", "entertainment","politics","sport","tech"]


for category in categories:
    count = 0
    articlepath = Path.joinpath(articlepath_folder, category)
    for entry in os.scandir(articlepath):
        text_file = open(entry, "r")
        raw_text = text_file.read()
        total_data.iloc[count, 2] = raw_text
        total_data.iloc[count, 0] = category
        text_file.close()

        count += 1

    count = 0
    summarypath = Path.joinpath(summarypath_folder, category)
    for entry in os.scandir(summarypath):
        text_file = open(entry, "r")
        raw_text = text_file.read()
        total_data.iloc[count, 1] = raw_text
        text_file.close()

        count += 1


#total_data.astype('string')

#print("nach Einlesen: ","\n",total_data)
#print("nach Einlesen: ","\n",total_data.iloc[1,2])
#print(total_data.dtypes)

## PREPROCESSING
# Remove Newline statements from Articles. Special case: first newline is the header which doesn't end with period.
# Separate Qoutes followed by other qoute

for i in range(10):
#Summary + Article_original
    for j in range(2):
        total_data.iloc[i, 1+j] = re.sub(r"\n\n", ". ", total_data.iloc[i, 1+j], 1)
        total_data.iloc[i, 1+j] = total_data.iloc[i, 1+j].replace("\n\n", " ")
        total_data.iloc[i, 1+j] = total_data.iloc[i, 1+j].replace("\n", " ")
        total_data.iloc[i, 1+j] = total_data.iloc[i, 1+j].replace('""', '" "')


# clean sentences: remove empty spaces and numbers
#Article_original -> Article_prep(rocessed)
        #total_data['Article_prep'] = total_data['Article_input'].str.lower()

    for j in range(1):
        #total_data.iloc[i, j + 3] = total_data.iloc[i,j + 2].str.lower()
        #total_data.iloc[i, j + 3] = re.sub(r'\W', ' ', total_data.iloc[i, j + 2]) #tut nichts?
        total_data.iloc[i, j + 3] = re.sub(r'\s+', ' ', total_data.iloc[i, j + 2])
        total_data.iloc[i, j + 3] = re.sub(r'\d+', '',total_data.iloc[i, j + 2])



## Preprocessing steps on one word basis: exclude stop words, lemmatize remaining words, remove punctuation useing RegexpTokenizer, all to lowercase
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        tokenizer = RegexpTokenizer(r'\w+')

        stop_words_deleted = []
        article_lemmatized =[]

        sentences = nltk.sent_tokenize(total_data.iloc[i, j + 3])

        for sentence in sentences:
            words = tokenizer.tokenize(sentence)
            words_lemmatized = []

            for word in words:
                if word in stop_words:
                    stop_words_deleted.append(word)
                else:
                    word = word.lower()
                    words_lemmatized.append(lemmatizer.lemmatize(word))
                    words_lemmatized.append(" ")

            words_lemmatized.append(". ")
            article_lemmatized.append("".join(words_lemmatized))
            #article_lemmatized.append(words_lemmatized)


        #total_data['Article_prep2'] = total_data['Article_prep2'].astype(object)
        #total_data.iat[1, 4] = article_lemmatized
        #total_data.loc[i, 'Article_prep2'] = article_lemmatized
        #print("Ergebnis: ", total_data.iloc[i, j + 4])

        total_data.iloc[i, j + 4] = "".join(article_lemmatized)
    #print( total_data.iloc[i, j + 4])


# print("nach Preprocessing: ","\n",total_data)
# print("Summary: ","\n",total_data.iloc[1,1])
# print("Article_input: ","\n",total_data.iloc[1,2])
# print("Article_prep: ","\n",total_data.iloc[1,3])
# print("Article_prep2: ","\n",total_data.iloc[1,4])


# for i in range(10):
#     total_data.iloc[i, 5] = Summarizer(total_data.iloc[i, 4])



# Saving Preprocessed Data
total_data.to_pickle("df_prep")

#os.remove("df_preprocessed")




