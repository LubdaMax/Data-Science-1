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
import pickle
import collections
from collections import Counter
import tensorflow as tf

tokenizer = RegexpTokenizer(r'\w+')

total_data = pd.read_pickle("df_prep")

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)


## CREATE VECTOR REPRESENTATION & APPLY METHOD

def bag_of_words(article_prep):
    """"returns dic with all words in input article as keys and count of words in article as values in descending order"""
    sentences = nltk.sent_tokenize(article_prep)
    print(sentences)
    word_frequency = {}
    tokenizer = RegexpTokenizer(r'\w+')

    for i in range(len(sentences)):
        words_per_sentence = tokenizer.tokenize(sentences[i])

        for word in words_per_sentence:
            if word not in word_frequency.keys():
                word_frequency[word] = 1
            else:
                word_frequency[word] += 1

    #word_frequency_sorted = OrderedDict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True))

    print("laenge word_frequency_dic: ",len(word_frequency))
    return word_frequency

def vector_representation(word_frequency, article):
    #use scikit-learn: count vectorizer instead (also to remove stop words)
    """"returns input article as matrix
    where each column represents a word out of the article and
    each rows a sentence indicating with 0/1 words in each sentence """

    sentences = nltk.sent_tokenize(article)
    print("#saetze:", len(sentences))
    word_vector = word_frequency.keys()
    article_vectors = []

    for sentence in sentences:
        words_per_sentence = tokenizer.tokenize(sentence)
        sentence_vector = []
        for word in word_frequency:
            if word in words_per_sentence:
                sentence_vector.append(1)
            else:
                sentence_vector.append(0)
        article_vectors.append(sentence_vector)

    article_matrix = pd.DataFrame(article_vectors, columns=word_vector)
    #print ("laenge article_vectors: ",len(article_vectors[1]))

    return article_matrix


def graph_representation (matrix_representation):
    """"generates similarity graph"""

    #normalize matrix
    normalized_matrix = TfidfTransformer().fit_transform(matrix_representation)
    similarity_graph = normalized_matrix * normalized_matrix.T
    #similarity_graph = similarity_graph.toarray()

    return similarity_graph



def pagerank (similarity_graph):
    """"applies PageRank, TextRank, LexRank"""
    #pagerank
    pagerank_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(pagerank_graph)
    scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)    #Ausgabe als Liste mit Tupels

    return scores_sorted



def Summarizer(article_prep):
    """"..."""""
    frequency_dic = bag_of_words(article_prep)
    graph_simple = vector_representation(frequency_dic, article_prep)
    similarity_graph = graph_representation(graph_simple)
    ranked = pagerank(similarity_graph)

    return ranked



for i in range (1):
    get_sentences_ranked= Summarizer(total_data.iloc[i, 4])
    ##sortiertes Dic als Liste  mit Tupeln [(1,0.3345),(8,0.2353)]

    get_original_summary = nltk.sent_tokenize(total_data.iloc[i,1])
    get_original_article = nltk.sent_tokenize(total_data.iloc[i,2])
    print("#Saetze Summary:",len(get_original_summary))
    print("#Saetze Artikel:",len(get_original_article))

    get_summary = []


    for i in range (len(get_original_summary)): ###summary nicht korrekt tokenized
        pos = get_sentences_ranked[i][0]
        get_summary.append(get_original_article[pos])


    print("original summary","\n", get_original_summary )
    print("pagerank summary: ", "\n", get_summary)






