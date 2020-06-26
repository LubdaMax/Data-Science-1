import pandas as pd
import os
import re
from pathlib import Path
import nltk
import pickle
# read data from the csv file (from the location it is stored)
# This is the original data provided by the author (link in report)

# This section reads in the large csv file and uses the first 10k 
# paragraphs to create a pikle file containing the raw data used for our
# project.
"""
rootpath = Path.cwd()
filepath = Path.joinpath(rootpath, r"Wikihow\wikihowSep.csv")
Data_full = pd.read_csv(filepath, engine='python')
Data_full = Data_full.astype(str)
Data = Data_full.iloc[0:10000,:]
rows, columns = Data.shape

filename = r"Wikihow\wiki_data_10k"
outfile = open(Path.joinpath(rootpath, filename) , 'wb')
pickle.dump(Data, outfile)
outfile.close()
"""

# Open the previously created file
rootpath = Path.cwd()
infile = open(Path.joinpath(rootpath, r"Wikihow\wiki_data_10k") , "rb")
Data = pickle.load(infile)
Data = Data.drop("sectionLabel", axis=1)
Data = Data.rename_axis("idx")
Data = Data.sort_values(["title", "idx"], ascending=[True, True])

# tokenize data
for i in range(Data.shape[0]):
    Data.iloc[i, 0] = nltk.sent_tokenize(Data.iloc[i, 0])
    Data.iloc[i, 1] = nltk.sent_tokenize(Data.iloc[i, 1])
    Data.iloc[i, 2] = nltk.sent_tokenize(Data.iloc[i, 2])

i = 0 #index for dataframe
j = -1 #index for dictionary
Article_dict = {}
prev_title = "asd"

# Creates Dictionary with articles as entries. Every article is a dataframe with one line per sentence.
while True:
    if Data.iloc[i, 3] != prev_title:
        j += 1
        prev_title = Data.iloc[i, 3]
        title = Data.iloc[i, 3]
        headline = Data.iloc[i, 1]
        overview = Data.iloc[i, 0]
        text = Data.iloc[i, 2]
        article = []
        in_summary = []

        # Titles and introductions are omitted as the cnn files lack these.
        #article.append(title)
        #in_summary.append(0)

        #for s in overview:
            #article.append(s)
            #in_summary.append(0)

        article.append(headline[0]) # Section headlines build the summaries
        in_summary.append(1)

        for s in text:
            article.append(s) 
            in_summary.append(0)

        Article_dict["Article{0}".format(j)] = pd.DataFrame({"sentence": article, "in_Summary": in_summary})
        i += 1

    if Data.iloc[i, 3] == prev_title:
    # the data is originally stored by paragraph. We now have to put togther the articles
    # by matching their titles.
        headline = Data.iloc[i, 1]
        text = Data.iloc[i, 2]
        article = []
        in_summary = []

        article.append(headline[0])
        in_summary.append(1)

        for s in text:
            article.append(s)
            in_summary.append(0)

        paragraph = pd.DataFrame({"sentence": article, "in_Summary": in_summary})
        Article_dict["Article{0}".format(j)] = Article_dict["Article{0}".format(j)].append(paragraph, ignore_index=True)
        i += 1

    if i >= Data.shape[0]:
        break

for i in range(len(Article_dict)):
    # Elimination of obstructive features in the data.
    Article_dict["Article{0}".format(i)] = Article_dict["Article{0}".format(i)][Article_dict["Article{0}".format(i)].sentence != "nan"]
    Article_dict["Article{0}".format(i)] = Article_dict["Article{0}".format(i)][Article_dict["Article{0}".format(i)].sentence != ";"]
    try:
        if type(int(Article_dict["Article{0}".format(i)].iloc[0, 0][-1])) == int:
            Article_dict["Article{0}".format(i)].iloc[0, 0] = Article_dict["Article{0}".format(i)].iloc[0, 0][:-1]
    except:
        pass
    Article_dict["Article{0}".format(i)].iloc[0, 0] = Article_dict["Article{0}".format(i)].iloc[0, 0] 
    for j in range(Article_dict["Article{0}".format(i)].shape[0]):
        Article_dict["Article{0}".format(i)].iloc[j, 0] = Article_dict["Article{0}".format(i)].iloc[j, 0].strip("\n")



# Export the file in pikle format
# For the use in Textrank we have also created files without the second column
# which contains the dependent variable.
# These have the tag "_1col"
filename = r"Wikihow\wiki_data_processed_no_overview_notitle_10k"
outfile = open(Path.joinpath(rootpath, filename) , 'wb')
pickle.dump(Article_dict, outfile)
outfile.close()

# Create a similar dicitonary containing the summaries-
Summary_dict = {}
for i in range(len(Article_dict)):
    summary = ""
    article = Article_dict["Article{0}".format(i)]
    for s in range(article.shape[0]):
        if article.iloc[s, 1] == 1:
            summary += article.iloc[s, 0]
            summary += " "
    summary = summary[:-1]
    Summary_dict["Summary{0}".format(i)] = summary

# Export the summaries         
summary_name = r"Wikihow\wiki_partial_summaries_10k"
summary_file = open(Path.joinpath(rootpath, summary_name), 'wb')
pickle.dump(Summary_dict, summary_file)
summary_file.close()

# Testcases
print(Summary_dict["Summary3"])

print(Article_dict["Article3"])