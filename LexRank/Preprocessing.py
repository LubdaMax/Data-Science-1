#import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
import pandas as pd
import os
import nltk
from nltk.stem import PorterStemmer
from pathlib import Path
import re
#import tensorflow_datasets as tfds
from collections import Counter


# Initialize the dataframe used to store Articles in col. 0
index = np.linspace(0, 510, 511)
total_data = pd.DataFrame(columns=["Article","Category"], index=index)
total_data = total_data.fillna(0)

# Relative path of dataset
rootpath = Path.cwd()
articlepath_folder = Path.joinpath(rootpath, r"Dataset 1 (BBC)\News Articles")
#summarypath_folder = Path.joinpath(rootpath, r"Dataset 1 (BBC)\Summaries")


# Insert data (ALL articles) from .txt files into the dataframe
#categories = ["business", "entertainment","politics","sport","tech"]

# Insert data (only business articles) from .txt files into the dataframe
categories = ["business"]

count = 0
for category in categories:
    #print("category: ", category)
    articlepath = Path.joinpath(articlepath_folder, category)
    #print("Articlepath: ", articlepath)
    for entry in os.scandir(articlepath):
        #print("entry: ",entry)
        text_file = open(entry, "r")
        raw_text = text_file.read()
        total_data.iloc[count, 0] = raw_text
        total_data.iloc[count, 1] = category
        text_file.close()

        count += 1

print(total_data.iloc[0,0])

# Remove Newline statements from Articles. Special case: first newline is the header which doesn't end with period.
# Separate Qoutes followed by other qoute
for i in range(510):
    for j in range(2):
        total_data.iloc[i, j] = re.sub(r"\n\n", ". ", total_data.iloc[i, j], 1)
        total_data.iloc[i, j] = total_data.iloc[i, j].replace("\n\n", " ")
        total_data.iloc[i, j] = total_data.iloc[i, j].replace("\n", " ")
        total_data.iloc[i, j] = total_data.iloc[i, j].replace('""', '" "')

# # Tokenize the data sentence by sentence
# for i in range(510):
#         total_data.iloc[i, 0] = nltk.sent_tokenize(str(total_data.iloc[i, 0]))

# # Tokenize the articles into words
# for i in range(510):
#         total_data.iloc[i, 0] = nltk.word_tokenize(str(total_data.iloc[i, 0]))

# # Tokenize the sentences into words
# for i in range(10):
#     for sentence in total_data.iloc[i, 0]:
#         sentence = nltk.word_tokenize(str(sentence))
#
# # #print(total_data.dtypes)
# print(total_data.iloc[0,0])


def bag_of_words (article):
    # tokenize each article into sentences
    article = nltk.sent_tokenize(article)

    word_frequency = {}
    word_frequency_stemmed = {}
    porter = PorterStemmer()


    # clean sentences: remove punctuation, empty spaces and convert to lower cases
    for i in range(len(article)):
        article[i] = article[i].lower()
        article[i] = re.sub(r'\W', ' ', article[i])
        article[i] = re.sub(r'\s+', ' ', article[i])


    # tokenize sentences into words and count frequency of words in dictionary (WITHOUT Stemming)
    # only add non stop words to the dictionary (remove stop words)
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
        stop_words_deleted = []

        words = nltk.word_tokenize(article[i])
        sentence_ = []

        for word in words:
            if word in stop_words:
                stop_words_deleted.append(word)
            elif word not in word_frequency.keys():
            #if word not in word_frequency.keys():
                word_frequency[word] = 1
                sentence_.append(word)
                sentence_.append(" ")
            else:
                word_frequency[word] += 1
                sentence_.append(word)
                sentence_.append(" ")

        #article[i] = "".join(sentence_)

    # tokenize sentences into words and count frequency of words in dictionary (WITH Stemming) - advantage: profit, profits don't appear twice in the dictionary
    # remove stop words
        words = nltk.word_tokenize(article[i])
        sentence_stemmed = []
        stop_words_stemmed_deleted = []

        for word in words:
            word = porter.stem(word)
            if word in stop_words:
                stop_words_stemmed_deleted.append(word)
            elif word not in word_frequency_stemmed.keys():
                word_frequency_stemmed[word] = 1
                sentence_stemmed.append(word)
                sentence_stemmed.append(" ")
            else:
                word_frequency_stemmed[word] += 1
                sentence_stemmed.append(word)
                sentence_stemmed.append(" ")

            # sentence_stemmed.append(word)
            # sentence_stemmed.append(" ")
        article[i] = "".join(sentence_stemmed)


    # # exclude stop words? use set of stop words predefined by nltk
    # from nltk.corpus import stopwords
    # stop_words = set(stopwords.words("english"))
    # stop_words_delete = []
    # for key in word_frequency_stemmed.keys():
    #     if key in stop_words:
    #         stop_words_delete.append(key)
    #
    # print("lenght of dict with stopwords: ", len(word_frequency_stemmed))
    # print("number stopwords to delete: ", len(stop_words_delete))
    #
    # for key in stop_words_delete:
    #     word_frequency_stemmed.pop(key)
    # print("lenght of dict without stopwords: ", len(word_frequency_stemmed))

    from collections import OrderedDict
    word_frequency_sorted = OrderedDict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True))
    word_frequency_stemmed_sorted = OrderedDict(sorted(word_frequency_stemmed.items(), key=lambda x: x[1], reverse=True))

    # # NEXT: convert sentences into vector representation
    # article_vectors = []
    # article = nltk.sent_tokenize(article)
    # for i in range(len(article)):
    #     sentence_words = nltk.word_tokenize(article[i])
    #     sentence_vector = []
    #     for word in word_frequency_stemmed.keys():
    #         if word in sentence_words:
    #             sentence_vector.append(1)
    #         else:
    #             sentence_vector.append(0)
    #     article_vectors.append(sentence_vector)




    # return stemmed sentences
    return article
    # return dic with stemmed words = keys, count of word = value, sorted
    #return word_frequency_stemmed_sorted
    #return article_vectors



print("vorher: ", total_data.iloc[0,0])
print("nachher: ",bag_of_words(total_data.iloc[0,0]))


# Tokenize the sentences into words
# for i in range(10):
#     total_data.iloc[i, 0] = bag_of_words(total_data.iloc[i, 0])
#     ## Problem: Dataframe Eintrag kann keine Liste sein?




