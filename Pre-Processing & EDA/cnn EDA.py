import numpy as np
import pandas as pd
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from pathlib import Path
import pickle



# unpickle preprocessed data (articles)
rootpath = Path.cwd()
openfile = open(Path.joinpath(rootpath, r"Pre-Processing & EDA\cnn_articles_dict"), 'rb')
cnn_article_dict = pickle.load(openfile)
openfile.close()

# unpickle preprocessed data (summaries)
rootpath = Path.cwd()
openfile = open(Path.joinpath(rootpath, r"Pre-Processing & EDA\cnn_summaries_dict"), 'rb')
cnn_summary_dict = pickle.load(openfile)
openfile.close()

# unpickle total dataframe
rootpath = Path.cwd()
openfile = open(Path.joinpath(rootpath, r"Pre-Processing & EDA\cnn_dataframe"), 'rb')
cnn_dataframe = pickle.load(openfile)
openfile.close()



# exploratory data analysis
sentences_per_article = []
words_per_article = []
words_per_sentenceA = []
for article in cnn_article_dict.keys():
    sentences_per_article.append(cnn_article_dict[article].shape[0])
for i in range(cnn_dataframe.shape[0]):
    for sentence in cnn_dataframe.iloc[i, 2]:
        words = word_tokenize(sentence)
        words_per_sentenceA.append(len(words))

sentencesA_total_number = sum(sentences_per_article)
wordsA_total_number = sum(words_per_sentenceA)

print("average of sentences per article: ",sentencesA_total_number/len(sentences_per_article))
print("average of words per article: ",wordsA_total_number/cnn_dataframe.shape[0])
print("average of words per sentence in article: ", wordsA_total_number/sentencesA_total_number)



sentences_per_summary = []
words_per_summary = []
words_per_sentenceS = []
for summary in cnn_summary_dict.keys():
    sentences_per_summary.append(cnn_summary_dict[summary].shape[0])
for i in range(cnn_dataframe.shape[0]):
    for sentence in cnn_dataframe.iloc[i, 3]:
        words = word_tokenize(sentence)
        words_per_sentenceS.append(len(words))

sentencesS_total_number = sum(sentences_per_summary)
wordsS_total_number = sum(words_per_sentenceS)

print("average of sentences per summary: ",sentencesS_total_number/len(sentences_per_summary))
print("average of words per summary: ",wordsS_total_number/cnn_dataframe.shape[0])
print("average of words per sentence in summary: ", wordsS_total_number/sentencesS_total_number)

# average ratio summary / article

print("average ratio of (words in summary / words in article)")
