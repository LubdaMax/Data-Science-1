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
from string import digits
import collections
from collections import Counter
import tensorflow as tf


## indicat what dataset is processed: cnn or wikihow
dataset = "cnn"
rootpath = Path.cwd()



if dataset == "cnn":

    # unpickle preprocessed data (articles)
    openfile = open(Path.joinpath(rootpath, r"Pre-Processing & EDA\cnn_articles_dict"), 'rb')
    data = pickle.load(openfile)
    openfile.close()

    # unpickle preprocessed data (summaries)
    openfile = open(Path.joinpath(rootpath, r"Pre-Processing & EDA\cnn_summaries_dict"), 'rb')
    summaries = pickle.load(openfile)
    openfile.close()

elif dataset == "wikihow":

    # unpickle preprocessed data (articles)
    openfile = open(Path.joinpath(rootpath, r"Pre-Processing & EDA\partial_data_processed_no_overview_notitle_1col"), 'rb')
    data = pickle.load(openfile)
    openfile.close()

    # unpickle preprocessed data (summaries)
    openfile = open(Path.joinpath(rootpath, r"Pre-Processing & EDA\wiki_partial_summaries"), 'rb')
    summaries = pickle.load(openfile)
    openfile.close()



def remove_punctuation(text):
    """
      function removes punctuation from given string parameter (text):
      !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
      """

    # for punctuation in string.punctuation:
    #     text = text.replace(punctuation,'')
    text = re.sub(r'[^\w\s]', '', text)

    return text


def remove_numbers(text):
    """" removes digital characters from string
    """
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)

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
    """ reduces every word in input string to a root form that exists in language
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
    """ reduces every word in input string to their root forms, word might not exist
        """
    porter = PorterStemmer()
    text_words_tokenized = word_tokenize(text)
    text_stemmed = []

    for word in text_words_tokenized:
        text_stemmed.append(porter.stem(word))
        text_stemmed.append(" ")
    text = "".join(text_stemmed)

    return text


def bag_of_words(article):
    """ returns a dictionary, where every key represents a word in teh text and the value the frequency of occurence
    :param article: article as a dataframe with one column, where each row represents 1 sentence
    :return: one dictionary : {wordA:frequency}
    """
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


def vector_representation(article, word_frequency):
    #use scikit-learn: count vectorizer instead (also to remove stop words)
    """" returns a vector for each sentence indicating for each word in the article
    whether it is contained in the sentence or not"""

    word_vector = word_frequency.keys()
    article_vectors = []
    for sentence in article:
        #print(sentence)
        words = word_tokenize(sentence)
        sentence_vector = []
        for word in word_frequency:
            if word in words:
                sentence_vector.append(1)
            else:
                sentence_vector.append(0)
        article_vectors.append(sentence_vector)

    return article_vectors


def graph_representation (vector_representation):
    """"generates similarity graph"""

    matrix_representation = pd.DataFrame(vector_representation)
    #normalize matrix
    normalized_matrix = TfidfTransformer().fit_transform(matrix_representation)
    similarity_graph = normalized_matrix * normalized_matrix.T


    return similarity_graph


def pagerank (similarity_graph):
    """" PageRank computes a ranking of the nodes in the graph G based on the structure of the incoming links.
        Scores are returned sorted.
        """
    pagerank_graph = nx.from_scipy_sparse_matrix(similarity_graph) #An adjacency matrix representation of the graph
    scores = nx.pagerank(pagerank_graph)
    scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)    #Ausgabe als Liste mit Tupels

    return scores_sorted


def generate_output_summ(ranked_sentences_dict, article_key, article_data, number_sentences):
    """ generates summary of a specified number of sentences
    :param ranked_sentences_dict: scores ranking all sentences in a text
    :param article_key: Identifier of article
    :param article_data: all articles as dataframe
    :param number_sentences:
    :return:
    """
    output_summary = []
    for i in range(number_sentences):
        s = ranked_sentences_dict[article_key][i][0]
        output_summary.append(article_data[article_key].iloc[s, 0])

    return output_summary



# PRE-PROCESSING II
# Remove punctuation, stop words, convert everything to lowercase, apply Lemmatization or Stemming
for article in data.keys():
    # df = data[article]
    # df.drop(columns=1)
    if dataset == "wikihow":
        data[article] = data[article].to_frame()
    data[article][1] = 0
    for sentence in range(len(data[article])):
        data[article].iloc[sentence, 1] = data[article].iloc[sentence, 0].lower()
        #data[article].iloc[sentence, 1] = remove_numbers(data[article].iloc[sentence, 1])
        data[article].iloc[sentence, 1] = remove_punctuation(data[article].iloc[sentence, 1])
        data[article].iloc[sentence, 1] = remove_stopwords(data[article].iloc[sentence, 1])
        data[article].iloc[sentence, 1] = apply_lemmatization(data[article].iloc[sentence, 1])
        #data[article].iloc[sentence, 1] = apply_stemming(data[article].iloc[sentence, 1])


# APPLY ALGORITHM
# Create Vector Representation of sentences // Create Similarity Graph and Apply Rank Method
article_frequency_dict = {}
article_vector_dict = {}
ranking_dict = {}
TRoutput_summ_dict = {}
count = 0


for key in data.keys():
    countStr = str(count)
    keyS = "Summary" + countStr
    article_frequency_dict[key] = bag_of_words(data[key])
    article_vector_dict[key] = vector_representation(data[key][1], article_frequency_dict[key])
    if not article_vector_dict[key] == []:
        ranking_dict[key] = pagerank((graph_representation(article_vector_dict[key])))
        try:
            TRoutput_summ_dict[keyS] = pd.DataFrame(generate_output_summ(ranking_dict, key, data, 3))
        except:
            TRoutput_summ_dict[keyS] = pd.DataFrame(generate_output_summ(ranking_dict, key, data, 2)) ##needed for longer summaries

    else:
        ranking_dict[key] = [(0,0)]
        TRoutput_summ_dict[keyS] = pd.DataFrame(generate_output_summ(ranking_dict, key, data, 0))
        print("Achtung, leerer Eintrag?")
    count += 1



# SAVE FILES WITH OUTPUTS
if dataset == "cnn":

    ## CNN / save summaries
    filename = r"TextRank\cnn_TRoutput_summ_dict_lemm_3sent_inclDigits"
    outfile = open(Path.joinpath(rootpath, filename), 'wb')
    pickle.dump(TRoutput_summ_dict, outfile)
    outfile.close()

    filename = r"TextRank\cnn_TRoutput_ranking_dict_lemm_3sent_inclDigits"
    outfile = open(Path.joinpath(rootpath, filename), 'wb')
    pickle.dump(ranking_dict, outfile)
    outfile.close()


elif dataset == "wikihow":

    ## WIKIHOW / save summaries
    filename = r"TextRank\wiki_TRoutput_summ_dict_lemm_3sent_inclDigits"
    outfile = open(Path.joinpath(rootpath, filename), 'wb')
    pickle.dump(TRoutput_summ_dict, outfile)
    outfile.close()

    filename = r"TextRank\wiki_TRoutput_ranking_dict_lemm_3sent_inclDigits"
    outfile = open(Path.joinpath(rootpath, filename), 'wb')
    pickle.dump(ranking_dict, outfile)
    outfile.close()


#Testcase

print("Artikel: ", data["Article3"])
print("Reference Summary: ", summaries["Summary3"])
print("Generated Summary: ", TRoutput_summ_dict["summary3"])






