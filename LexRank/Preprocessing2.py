import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pathlib import Path
import re
from collections import Counter

# Initialize the dataframe used to store Articles in col. 0         ######## TO DO:  fit Array size to article categories
index = np.linspace(0, 510, 511)
total_data = pd.DataFrame(columns=["Category","Summary","Article_input","Article_prep","Article_prep2","Summary_output"], index=index, dtype = string)
total_data = total_data.fillna(0)

# Relative path of dataset
rootpath = Path.cwd()
articlepath_folder = Path.joinpath(rootpath, r"Dataset 1 (BBC)\News Articles")
#summarypath_folder = Path.joinpath(rootpath, r"Dataset 1 (BBC)\Summaries")


# Insert data from .txt files into the dataframe
categories = ["business"]
#categories = ["business", "entertainment","politics","sport","tech"]

count = 0
for category in categories:
    articlepath = Path.joinpath(articlepath_folder, category)
    for entry in os.scandir(articlepath):
        text_file = open(entry, "r")
        raw_text = text_file.read()
        total_data.iloc[count, 2] = raw_text
        total_data.iloc[count, 0] = category
        text_file.close()

        count += 1


## PREPROCESSING
# Remove Newline statements from Articles. Special case: first newline is the header which doesn't end with period.
# Separate Qoutes followed by other qoute
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

for i in range(10):
#Summary + Article_original
    for j in range(2):
        total_data.iloc[i, 1+j] = re.sub(r"\n\n", ". ", total_data.iloc[i, 1+j], 1)
        total_data.iloc[i, 1+j] = total_data.iloc[i, 1+j].replace("\n\n", " ")
        total_data.iloc[i, 1+j] = total_data.iloc[i, 1+j].replace("\n", " ")
        total_data.iloc[i, 1+j] = total_data.iloc[i, 1+j].replace('""', '" "')

# clean sentences: remove punctuation, empty spaces and convert to lower cases
#Article_original -> Article_prep(rocessed)
    for j in range (1):
        total_data.iloc[i, j + 3] = total_data.iloc[i,j + 2].str.lower()
        total_data.iloc[i, j + 3] = re.sub(r'\W', ' ', total_data.iloc[i, j + 2])
        total_data.iloc[i, j + 3] = re.sub(r'\s+', ' ', total_data.iloc[i, j + 2])

## Preprocessing steps on one word basis: exclude stop words, lemmatize remaining words
        sentences = nltk.sent_tokenize(total_data.iloc[i, j + 3])
        words = nltk.word_tokenize(sentences)
        stop_words_deleted = []
        words_lemmatized = []

        for word in words:
            if word in stop_words:
                stop_words_deleted.append(word)
            else:
                words_lemmatized.append(lemmatizer.lemmatize(word))
                words_lemmatized.append(" ")

        total_data.iloc[i, j + 4] = "".join(words_lemmatized)


## CREATE VECTOR REPRESENTATION & APPLY METHOD

for i in range(10):
    total_data.iloc[i, 5] = Summarizer(total_data.iloc[i, 4])


def Summarizer(article_prep)
    """"..."""""

    return





def bag_of_words(article_prep):
    """"returns dic with all words in input article as keys and count of words in article as values in descending order"""
    sentences = nltk.sent_tokenize(article)
    word_frequency = {}

    for i in range(len(sentences)):
        words_per_sentence = nltk.word_tokenize(sentences[i])

        for word in words_per_sentence:
            if word not in word_frequency.keys():
                word_frequency[word] = 1
            else:
                word_frequency[word] += 1

    word_frequency_sorted = OrderedDict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True))

    return word_frequency_sorted

def vector_representation(word_frequency_sorted, article):
    """"returns input article as matrix where rows are sentences and columns represent words .. """

    article_vectors = []
    for sentence in article:
        words_per_sentence = nltk.word_tokenize(sentence)
        sentence_vector = []
        for word in most_freq:
            if word in words_per_sentence:
                sentence_vector.append(1)
            else:
                sentence_vector.append(0)
        article_vectors.append(sentence_vector)

    article_matrix = pd.array(article_vectors, dtype = string)

    return article_matrix


def graph_representation (matrix_representation):
    """"generates similarity graph"""

    from sklearn.feature_extraction.text import TfidfTransformer
    >> > normalized_matrix = TfidfTransformer().fit_transform(matrix_representation)

    >> > similarity_graph = normalized_matrix * normalize_matrix.T
    >> > similarity_graph.toarray()
    " This is a mirrored matrix, where both the rows and columns correspond to sentences, and the elements describe how similar the two sentences are. Scores of 1 mean that the sentences are exactly the same, while scores of 0 mean the sentences have no overlap."

    return similarity_graph



def rank (similarity_graph)
    """"applies PageRank, TextRank, LexRank"""
    return summary



