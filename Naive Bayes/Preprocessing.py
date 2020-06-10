import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import nltk

import tensorflow_datasets as tfds

#data = tfds.load("cnn_dailymail") 

index = np.linspace(0, 509, 510)
business_data = pd.DataFrame(columns=["Article", "Summary"], index=index)
business_data = business_data.fillna(0)

business_article_dir = r'C:\Users\user\Documents\Uni\Winfo\1. Semester\DataScience1\Summarizer\24984_32267_bundle_archive\BBC News Summary\News Articles\business'
business_summary_dir = r'C:\Users\user\Documents\Uni\Winfo\1. Semester\DataScience1\Summarizer\24984_32267_bundle_archive\BBC News Summary\Summaries\business'


count = 0
for entry in os.scandir(business_article_dir):
    text_file = open(entry, "r")
    raw_text = text_file.read()
    business_data.iloc[count, 0] = raw_text
    text_file.close()

    count += 1

count = 0
for entry in os.scandir(business_summary_dir):
    text_file = open(entry, "r")
    raw_text = text_file.read()
    business_data.iloc[count, 1] = raw_text
    text_file.close()
    
    count += 1

#print(business_data.head())
#print(business_data.tail())

for i in range(510):
    for j in range(2):
        business_data.iloc[i, j] = nltk.sent_tokenize(str(business_data.iloc[i, j]))
"""   
for i in range(510):
    for sentence in business_data.iloc[i, 0]:
        if sentence in business_data.iloc[i, 1]:
            business_data.iloc[i, 0] = sentence.replace(sentence, (1, sentence))
        else:
            business_data.iloc[i, 0] = sentence.replace(sentence, (0, sentence))         
"""

print(business_data.head())
print(business_data.tail())
print(business_data.iloc[0, 1])
