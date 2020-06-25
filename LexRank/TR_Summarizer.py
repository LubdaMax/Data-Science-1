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


desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)

## CNN
# unpickle preprocessed data (articles)
os.chdir("C:/Users/Leni/Google Drive/00_Studium/01_Master WI Goethe/01_Veranstaltungen/SS20_DS_Data Science 1/DS Project/NLP _Text Summarizer/")
rootpath = Path.cwd()
openfile = open(Path.joinpath(rootpath, r"Pre-Processing & EDA\cnn_articles_dict"), 'rb')
data = pickle.load(openfile)
openfile.close()

# unpickle preprocessed data (summaries)
#os.chdir("C:/Users/Leni/Google Drive/00_Studium/01_Master WI Goethe/01_Veranstaltungen/SS20_DS_Data Science 1/DS Project/NLP _Text Summarizer/")
rootpath = Path.cwd()
openfile = open(Path.joinpath(rootpath, r"Pre-Processing & EDA\cnn_summaries_dict"), 'rb')
summaries = pickle.load(openfile)
openfile.close()

#
# ## WIKIHOW
# os.chdir("C:/Users/Leni/Google Drive/00_Studium/01_Master WI Goethe/01_Veranstaltungen/SS20_DS_Data Science 1/DS Project/NLP _Text Summarizer/")
# rootpath = Path.cwd()
#
# # unpickle preprocessed data (articles)
# openfile = open(Path.joinpath(rootpath, r"Wikihow\partial_data_processed_no_overview"), 'rb')
# data = pickle.load(openfile)
# openfile.close()
#
# # unpickle preprocessed data (summaries)
# openfile = open(Path.joinpath(rootpath, "Wikihow\wiki_partial_summaries"), 'rb')
# summaries = pickle.load(openfile)
# openfile.close()
#
# for key in data.keys():
#     if data[key].iloc[0,0]==0:
#         print(key, data[key])




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
    # df = data[article]
    # df.drop(columns=1)
    data[article][1] = 0
    for sentence in range(len(data[article])):
        data[article].iloc[sentence, 1] = data[article].iloc[sentence, 0].lower()
        data[article].iloc[sentence, 1] = remove_numbers(data[article].iloc[sentence, 1])
        data[article].iloc[sentence, 1] = remove_punctuation(data[article].iloc[sentence, 1])
        data[article].iloc[sentence, 1] = remove_stopwords(data[article].iloc[sentence, 1])
        #data[article].iloc[sentence, 1] = apply_lemmatization(data[article].iloc[sentence, 0])
        data[article].iloc[sentence, 1] = apply_stemming(data[article].iloc[sentence, 1])


def bag_of_words(article):
    """

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
    """"returns input article as matrix
    where each column represents a word out of the article and
    each rows a sentence indicating with 0/1 words in each sentence """

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
    #article_matrix = pd.DataFrame(article_vectors, columns=word_vector)

    return article_vectors

def graph_representation (vector_representation):
    """"generates similarity graph"""

    matrix_representation = pd.DataFrame(vector_representation)
    #normalize matrix
    normalized_matrix = TfidfTransformer().fit_transform(matrix_representation)
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


def generate_output_summ(ranked_sentences_dict, article_key, article_data, number_sentences):
    output_summary = []
    for i in range(number_sentences):
        s = ranked_sentences_dict[article_key][i][0]
        output_summary.append(article_data[article_key].iloc[s, 0])

    return output_summary

# Create Vector Representation of sentences // Create Similarity Graph and Apply Rank Method
article_frequency_dict = {}
article_vector_dict = {}
ranking_dict = {}
TRoutput_summ_dict = {}
count = 0


for key in data.keys():
    keyS = "Summary" + str(count)
    article_frequency_dict[key] = bag_of_words(data[key])
    article_vector_dict[key] = vector_representation(data[key][1], article_frequency_dict[key])
    if not article_vector_dict[key] == []:
        ranking_dict[key] = pagerank((graph_representation(article_vector_dict[key])))
        TRoutput_summ_dict[keyS] = pd.DataFrame(generate_output_summ(ranking_dict, key, data, 3))
    else:
        ranking_dict[key] = [(0,0)]
        TRoutput_summ_dict[keyS] = pd.DataFrame(generate_output_summ(ranking_dict, key, data, 0))
        print("Achtung, leerer Eintrag?")
        print(ranking_dict[key])
        print(TRoutput_summ_dict[keyS])
        print(data[key])

    count += count

#print(ranking_dict)
#print(TRoutput_summ_dict)



os.chdir("C:/Users/Leni/Google Drive/00_Studium/01_Master WI Goethe/01_Veranstaltungen/SS20_DS_Data Science 1/DS Project/NLP _Text Summarizer/")
rootpath = Path.cwd()

## CNN / save summaries
filename = r"LexRank\cnn_TRoutput_summ_dict"
outfile = open(Path.joinpath(rootpath, filename), 'wb')
pickle.dump(TRoutput_summ_dict, outfile)
outfile.close()

filename = r"LexRank\cnn_TRoutput_ranking_dict"
outfile = open(Path.joinpath(rootpath, filename), 'wb')
pickle.dump(ranking_dict, outfile)
outfile.close()


# ## WIKIHOW / save summaries
# filename = r"LexRank\wiki_TRoutput_summ_dict"
# outfile = open(Path.joinpath(rootpath, filename), 'wb')
# pickle.dump(TRoutput_summ_dict, outfile)
# outfile.close()
#
# filename = r"LexRank\wiki_TRoutput_ranking_dict"
# outfile = open(Path.joinpath(rootpath, filename), 'wb')
# pickle.dump(ranking_dict, outfile)
# outfile.close()





### CALLING TR on any Text:
def use_TR(text, len_summary):
    """

    :param text:
    :return:
    """

    text_tokenized = nltk.sent_tokenize(text)
    if len(text_tokenized) < len_summary:
        return "Fehler bei der Eingabe, Zusammenfassung soll länger als Text sein"
    else:

        for i in range(len(text_tokenized)):
            text_tokenized[i] = text_tokenized[i].lower()
            text_tokenized[i] = remove_stopwords(text_tokenized[i])
            text_tokenized[i] = remove_punctuation(text_tokenized[i])
            #text_tokenized[i] = apply_lemmatization(text_tokenized[i])
            text_tokenized[i] = apply_stemming(text_tokenized[i])

        text_df = pd.DataFrame(text_tokenized)

        word_frequency = bag_of_words(text_df)
        text_vector = vector_representation(text_df, word_frequency)
        sentence_ranking = pagerank((graph_representation(text_vector)))

        summary = []

        count = 0
        for i in sentence_ranking.keys():
            if count == len_summary:
                break
            else:
                summary.append(text_df[0,i])
                summary.append(" ")
                count += 1

        output_summary = " ".join(summary)

    return output_summary




