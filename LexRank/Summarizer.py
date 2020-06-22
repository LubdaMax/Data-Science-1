import numpy as np
import pandas as pd
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfTransformer
import networkx as nx
import pickle
import string
import collections
from collections import Counter
import tensorflow as tf


# unpickle preprocessed data (articles)
#os.chdir("C:/Users/Leni/Google Drive/00_Studium/01_Master WI Goethe/01_Veranstaltungen/SS20_DS_Data Science 1/DS Project/NLP _Text Summarizer/")
rootpath = Path.cwd()
openfile = open(Path.joinpath(rootpath, r"cnn_articles_dict"), 'rb')
data = pickle.load(openfile)
openfile.close()

# unpickle preprocessed data (summaries)
#os.chdir("C:/Users/Leni/Google Drive/00_Studium/01_Master WI Goethe/01_Veranstaltungen/SS20_DS_Data Science 1/DS Project/NLP _Text Summarizer/")
rootpath = Path.cwd()
openfile = open(Path.joinpath(rootpath, r"cnn_summaries_dict"), 'rb')
summaries = pickle.load(openfile)
openfile.close()



def remove_punctuation(text):
    """
      function removes punctuation from given string parameter (text):
      !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
      """

    for punctuation in string.punctuation:
        text = text.replace(punctuation,'')
    #text = re.sub(r'[^\w\s]', '', text)

    return text


def remove_stopwords(text):
    """
        function removes english stopwords from given parameter (text):
        "you've", 'any', "shan't", 'a', "don't", 'and', 'during', 'will', 'been', 'won', 'can', 'mustn',
        'which', 'what', 'once', 'themselves', 'himself', 'not', 'on', 'his', 'yourself', 'myself', 'here',
        "doesn't", 'hadn', 'for', 'why', 'him', 'd', 'off', 's', 'some', 'of', 'shouldn', 'o', 'same', 'now', 'to',
        'itself', 'yours', 'isn', 'them', 'who', 'its', 'no', 'above', 'out', 'all', 'i', 'he', 'shan', 'only', 're',
        'through', 'before', 'these', "mightn't", 'had', 'other', 'in', 'than', 'at', 'most', 'have', 'while', 'theirs',
        'as', "weren't", "she's", 'did', 'just', 'the', 'don', 'between', 'when', 'until', 'hasn', "wasn't", 'having',
        'more', "shouldn't", 'too', 'but', 'own', "aren't", "you'd", 'me', 've', 'ain', 'couldn', 'ourselves', 'against',
        'hers', "wouldn't", 'were', 'she', 'by', 'so', 'yourselves', 'further', 'haven', 'was', 'about', 'am', 'needn',
        'it', 'be', 'from', "hasn't", 'after', 'being', 'aren', 'herself', 'wasn', 'does', 'very', 'or', 'they', 'over',
        "should've", 'is', 'again', 'y', 'do', "needn't", 'with', 'has', 'below', "you're", 'how', 'because', 'my',
        'down', 'up', 'that', 'into', 'nor', "you'll", "didn't", 'our', 'where', 'didn', 'then', 'm', 'there', "hadn't",
        'mightn', 'your', "that'll", 'whom', 'an', "haven't", 'ma', "mustn't", 'doesn', 'we', 'this', 'those', 'you',
        'weren', 'if', 'should', 'wouldn', "it's", 'ours', 'such', 'their', 'under', 'each', 't', 'are', 'her', 'few',
        "couldn't", "isn't", 'both', 'doing', 'll', "won't"}

        """

    stop_words = set(stopwords.words("english"))
    text_words_tokenized = word_tokenize(text)
    text_without_stopwords = []

    for word in text_words_tokenized:
        if word not in stop_words:
            text_without_stopwords.append(word)
            text_without_stopwords.append(" ")

    text = "".join(text_without_stopwords)

    return text

def apply_lemmatization(text):
    """
        """
    lemmatizer = WordNetLemmatizer()
    text_words_tokenized = word_tokenize(text)
    text_lemmatized = []

    for word in text_words_tokenized:
        text_lemmatized.append(lemmatizer.lemmatize(word))
        text_lemmatized.append(" ")
    text = "".join(text_lemmatized)

    return text

def apply_stemming(text):
    """
        """
    porter = PorterStemmer()
    text_words_tokenized = word_tokenize(text)
    text_stemmed = []

    for word in text_words_tokenized:
        text_stemmed.append(porter.stem(word))
        text_stemmed.append(" ")
    text = "".join(text_stemmed)

    return text


# Remove punctuation, stop words, convert everything to lowercase, apply Lemmatization or Stemming
for article in data.keys():
    data[article][1] = 0
    for sentence in range(len(data[article])):
        data[article].iloc[sentence, 1] = data[article].iloc[sentence, 0].lower()
        data[article].iloc[sentence, 1] = remove_stopwords(data[article].iloc[sentence, 0])
        data[article].iloc[sentence, 1] = remove_punctuation(data[article].iloc[sentence, 0])
        #data[article].iloc[sentence, 1] = apply_lemmatization(data[article].iloc[sentence, 0])
        data[article].iloc[sentence, 1] = apply_stemming(data[article].iloc[sentence, 0])


def bag_of_words(article):
    word_frequency = {}
    for sentence in range(article.shape[0]):
        words = word_tokenize(article.iloc[sentence, 1])
        for w in words:
            if w not in word_frequency.keys():
                word_frequency[w] = 1
            else:
                word_frequency[w] += 1
    #word_frequency_sorted = OrderedDict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True))

    return word_frequency

def vector_representation(article,word_frequency):
    #use scikit-learn: count vectorizer instead (also to remove stop words)
    """"returns input article as matrix
    where each column represents a word out of the article and
    each rows a sentence indicating with 0/1 words in each sentence """

    word_vector = word_frequency.keys()
    article_vectors = []
    for sentence in article:
        words = word_tokenize(sentence)
        sentence_vector = []
        for word in word_frequency:
            if word in words:
                sentence_vector.append(1)
            else:
                sentence_vector.append(0)
        article_vectors.append(sentence_vector)
    #article_matrix = pd.DataFrame(article_vectors, columns=word_vector)

    return article_vectors

def graph_representation (vector_representation):
    """"generates similarity graph"""
    matrix = pd.DataFrame(vector_representation)
    #normalize matrix
    normalized_matrix = TfidfTransformer().fit_transform(matrix)
    similarity_graph = normalized_matrix * normalized_matrix.T
    #similarity_graph = similarity_graph.toarray()

    return similarity_graph

def pagerank (similarity_graph):
    """"
        """
    pagerank_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(pagerank_graph)
    scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)    #Ausgabe als Liste mit Tupels

    return scores_sorted

# Create Vector Representation of sentences // Create Similarity Graph and Apply Rank Method
article_frequency_dict = {}
article_vector_dict = {}
ranking_dict = {}
cnnTRoutput_summ_dict = {}
count = 0

#print(data["Article0"])
#print(data.keys())
for key in data.keys():
    #print(article)
    keyS = "Summary" + str(count)
    article_frequency_dict[key] = bag_of_words(data[key])
    #print(article_frequency_dict[key])
    article_vector_dict[key] = vector_representation(data[key][1], article_frequency_dict[key])
    #print(data[key][1])
    #print(vector_representation(data[key][1], article_frequency_dict[key]))
    ranking_dict[key] = pagerank((graph_representation(article_vector_dict[key])))
    #print(graph_representation(article_vector_dict[key]))

    output_summary = []
    for i in range(summaries[keyS].shape[0]):
        s = ranking_dict[key][i][0]
        output_summary.append(data[key].iloc[s, 0])

    cnnTRoutput_summ_dict[keyS] = pd.DataFrame(output_summary)
    count += count




# filename = r"cnnTRoutput_summ_dict"
# outfile = open(Path.joinpath(rootpath, filename), 'wb')
# pickle.dump(cnnTRoutput_summ_dict, outfile)
# outfile.close()




## CREATE VECTOR REPRESENTATION & APPLY METHOD

# def vector_representation(word_frequency, article):
#     #use scikit-learn: count vectorizer instead (also to remove stop words)
#     """"returns input article as matrix
#     where each column represents a word out of the article and
#     each rows a sentence indicating with 0/1 words in each sentence """
#
#     sentences = nltk.sent_tokenize(article)
#     print("#saetze:", len(sentences))
#     word_vector = word_frequency.keys()
#     article_vectors = []
#
#     for sentence in sentences:
#         words_per_sentence = tokenizer.tokenize(sentence)
#         sentence_vector = []
#         for word in word_frequency:
#             if word in words_per_sentence:
#                 sentence_vector.append(1)
#             else:
#                 sentence_vector.append(0)
#         article_vectors.append(sentence_vector)
#
#     article_matrix = pd.DataFrame(article_vectors, columns=word_vector)
#     #print ("laenge article_vectors: ",len(article_vectors[1]))
#
#     return article_matrix
#
#
# def graph_representation (matrix_representation):
#     """"generates similarity graph"""
#
#     #normalize matrix
#     normalized_matrix = TfidfTransformer().fit_transform(matrix_representation)
#     similarity_graph = normalized_matrix * normalized_matrix.T
#     #similarity_graph = similarity_graph.toarray()
#
#     return similarity_graph
#
#
#
# def pagerank (similarity_graph):
#     """"applies PageRank, TextRank, LexRank"""
#     #pagerank
#     pagerank_graph = nx.from_scipy_sparse_matrix(similarity_graph)
#     scores = nx.pagerank(pagerank_graph)
#     scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)    #Ausgabe als Liste mit Tupels
#
#     return scores_sorted
#
#
#
# def Summarizer(article_prep):
#     """"..."""""
#     frequency_dic = bag_of_words(article_prep)
#     graph_simple = vector_representation(frequency_dic, article_prep)
#     similarity_graph = graph_representation(graph_simple)
#     ranked = pagerank(similarity_graph)
#
#     return ranked
#


# for i in range (1):
#     get_sentences_ranked= Summarizer(total_data.iloc[i, 4])
#     ##sortiertes Dic als Liste  mit Tupeln [(1,0.3345),(8,0.2353)]
#
#     get_original_summary = nltk.sent_tokenize(total_data.iloc[i,1])
#     get_original_article = nltk.sent_tokenize(total_data.iloc[i,2])
#     print("#Saetze Summary:",len(get_original_summary))
#     print("#Saetze Artikel:",len(get_original_article))
#
#     get_summary = []
#
#
#     for i in range (len(get_original_summary)): ###summary nicht korrekt tokenized
#         pos = get_sentences_ranked[i][0]
#         get_summary.append(get_original_article[pos])
#
#
#     print("original summary","\n", get_original_summary )
#     print("pagerank summary: ", "\n", get_summary)






