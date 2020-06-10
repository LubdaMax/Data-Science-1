import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import nltk
from pathlib import Path
import re
import tensorflow_datasets as tfds

# Initialize the dataframe used to store Articles in col. 0 and Summaries in col. 1
index = np.linspace(0, 509, 510)
business_data = pd.DataFrame(columns=["Article", "Summary"], index=index)
business_data = business_data.fillna(0)

# Relative path of dataset
rootpath = Path.cwd()
articlepath = Path.joinpath(rootpath, r"Dataset 1 (BBC)\News Articles\business")
summarypath = Path.joinpath(rootpath, r"Dataset 1 (BBC)\Summaries\business")


# Insert data from .txt files into the dataframe
count = 0
for entry in os.scandir(articlepath):
    text_file = open(entry, "r")
    raw_text = text_file.read()
    business_data.iloc[count, 0] = raw_text
    text_file.close()

    count += 1

count = 0
for entry in os.scandir(summarypath):
    text_file = open(entry, "r")
    raw_text = text_file.read()
    business_data.iloc[count, 1] = raw_text
    text_file.close()
    
    count += 1

# Add spaces after the dots in the summaries (They are missing in the original summaries)
for i in range(510):
    business_data.iloc[i, 1] = re.sub(r"\D\.", ". ", business_data.iloc[i, 1], 0)


# Tokenize the data sentence by sentence
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
