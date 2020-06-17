import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import nltk
from pathlib import Path
import re
import tensorflow_datasets as tfds
import pickle


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
# Fixed wrong space placement in quoutes.
#for i in range(510):
#    business_data.iloc[i, 1] = re.sub(r"\D\.", ". ", business_data.iloc[i, 1], 0) regex don't work...

for i in range(510):
    business_data.iloc[i, 1] = business_data.iloc[i, 1].replace(".", ". ")
    business_data.iloc[i, 1] = business_data.iloc[i, 1].replace('. "', '."')
    business_data.iloc[i, 1] = business_data.iloc[i, 1].replace('. . . ', '...')

replacement_dict = {"0. ": "0.", "1. ": "1.", "2. ": "2.", "3. ": "3.", "4. ": "4.", "5. ": "5.", "6. ": "6.", "7. ": "7.", "8. ": "8.", "9. ": "9."}

for k in range(510):
    for i, j in replacement_dict.items():
        business_data.iloc[k, 1] = business_data.iloc[k, 1].replace(i, j)

# Remove Newline statements from Articles. Special case: first newline is the header which doesn't end with period.
# Separate Qoutes followed by other qoute
for i in range(510):
    for j in range(2):
        business_data.iloc[i, j] = re.sub(r"\n\n", ". ", business_data.iloc[i, j], 1)
        business_data.iloc[i, j] = business_data.iloc[i, j].replace("\n\n", " ")
        business_data.iloc[i, j] = business_data.iloc[i, j].replace("\n", " ")
        business_data.iloc[i, j] = business_data.iloc[i, j].replace('""', '" "')
        business_data.iloc[i, j] = business_data.iloc[i, j].replace("\\", "")
        #business_data.iloc[i, j] = business_data.iloc[i, j].replace('"', '')

# Tokenize the data sentence by sentence
for i in range(510):
    for j in range(2):
        business_data.iloc[i, j] = nltk.sent_tokenize(business_data.iloc[i, j])


# Create one dataframe per doc
subframes = {}
for doc in range(510):
    subframes["Article{0}".format(doc)] = pd.DataFrame(data=None, index=np.linspace(0, len(business_data.iloc[doc, 0]) - 1, len(business_data.iloc[doc, 0])), 
    columns=["Sentence", "in_Summary"])

    for i in range(len(business_data.iloc[doc, 0])):
        subframes["Article{0}".format(doc)].iloc[i, 0] = business_data.iloc[doc, 0][i]
        if business_data.iloc[doc, 0][i] in business_data.iloc[doc, 1]:
            subframes["Article{0}".format(doc)].iloc[i, 1] = 1
        else:
            subframes["Article{0}".format(doc)].iloc[i, 1] = 0


# Testcase and data exclusion
Testscores = np.zeros(510)
subframes_working = {}
count = 0
for i in range(510):    
    Testscores[i] = abs(subframes["Article{0}".format(i)]["in_Summary"].sum() - len(business_data.iloc[i, 1]))

    if Testscores[i] == 0:
        subframes_working["Article{0}".format(count)] = subframes["Article{0}".format(i)]
        count += 1


working_cases = 0
for i in Testscores:
    if i == 0:
        working_cases += 1
working_percent = working_cases / 510

#print(business_data.iloc[276,0], business_data.iloc[276,1])
#print(subframes["Article276"], len(business_data.iloc[276, 0]))
#print(Testscores.max(), Testscores.argmax(), Testscores.sum())
#print("working cases: ", working_cases, "Percent: ", working_percent)
#print(len(subframes_working))

# Saving Preprocessed Data
filename = "data_dependent_var"
outfile = open(filename, 'wb')
pickle.dump(subframes_working, outfile)
outfile.close()