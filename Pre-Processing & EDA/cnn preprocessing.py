from pathlib import Path
import os
import pickle
import shutil
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


# # select 1000 files randomly and save to another folder
# cnn_source_path = r""
# cnn_save_path = r""

# files = os.listdir(cnn_source_path)
# random_files = np.random.choice(files, 1000)
# # save 1000 randomly selected files to shared folder (github)
# for file in random_files:
#     source_file = os.path.join(cnn_source_path, file)
#     shutil.copy(source_file, cnn_save_path)


# Relative path of dataset
rootpath = Path.cwd()
cnn_path = Path.joinpath(rootpath, r"CNN")

index = np.linspace(0, 999, 1000)
cnn_data = pd.DataFrame(columns=["filename","text_raw", "text_prep", "summary"], index = index.astype(int))
cnn_data = cnn_data.fillna("nan")


# save articles from txt files to dataframe
count = 0
for entry in os.scandir(cnn_path):
    if count != 1000:
        text_file = open(entry, "r", encoding="utf8")
        raw_text = text_file.read()
        cnn_data.iloc[count, 1] = raw_text
        cnn_data.iloc[count, 0] = str(entry)
        text_file.close()

        count += 1
    else:
        break



# make texts usable for tokenization etc, remove unnecessary information
to_drop = []

for i in range(cnn_data.shape[0]):
    cnn_data.iloc[i, 2] = re.sub("\n\n",". ",cnn_data.iloc[i,1]) ##remove new line symbol
    cnn_data.iloc[i, 2] = cnn_data.iloc[i, 2].replace("..",".")
    cnn_data.iloc[i, 2] = nltk.sent_tokenize(cnn_data.iloc[i, 2])
    if cnn_data.iloc[i, 2] != []:
        cnn_data.iloc[i, 2][0] = re.sub('^.*?-- ',"",cnn_data.iloc[i, 2][0]) #removes: (CNN) --
        cnn_data.iloc[i, 2][0] = re.sub('^.*?\(CNN\)', "", cnn_data.iloc[i, 2][0])  # removes: (CNN)
    else:
        to_drop.append(i)



# extract highlighted sentenced from text as summary of the text
    sentences = []
    summary = []
    article = cnn_data.iloc[i, 2]
    count = 0

    for s in range(len(article)):
        #print("s:", s)
        if article[s] != '@highlight.':
            sentences.append(article[s])
            #print(article[s])
        elif article[s] == '@highlight.':
            count = s
            break

    for j in range(count,len(article)):
        #print("j:", j)
        if article[j] != '@highlight.':
            #print(summary)
            summary.append(article[j])

    cnn_data.iloc[i, 2] = sentences
    cnn_data.iloc[i, 3] = summary

    if "(CNN)" not in (cnn_data.iloc[i,1]) and i not in to_drop:
        to_drop.append(i)

   # print('summary ',cnn_data.iloc[i, 3])
    #print('article ', cnn_data.iloc[i, 2])


# drop empty texts
cnn_data.drop(cnn_data.index[to_drop], inplace=True)
cnn_data = cnn_data.reset_index(drop=True)



## save Articles and Summaries in Dictionary, having a dataframe as item which contains one row per sentence
cnn_article_dict = {}
for i in range(cnn_data.shape[0]):
    key = "Article" + str(i)
    cnn_article_dict[key] = pd.DataFrame(cnn_data.iloc[i, 2])



cnn_summary_dict = {}

for i in range(cnn_data.shape[0]):
    key = "Summary" + str(i)
    cnn_summary_dict[key] = pd.DataFrame(cnn_data.iloc[i, 3])


cnn_summary_dataframe = cnn_data.iloc[i, 3]




# save outputs
filename = r"Pre-Processing & EDA\cnn_articles_dict"
outfile = open(Path.joinpath(rootpath, filename), 'wb')
pickle.dump(cnn_article_dict, outfile)
outfile.close()

filename = r"Pre-Processing & EDA\cnn_summaries_dict"
outfile = open(Path.joinpath(rootpath, filename), 'wb')
pickle.dump(cnn_summary_dict, outfile)
outfile.close()

filename = r"Pre-Processing & EDA\cnn_data_dataframe"
outfile = open(Path.joinpath(rootpath, filename), 'wb')
pickle.dump(cnn_data, outfile)
outfile.close()

filename = r"Pre-Processing & EDA\cnn_summary_dataframe"
outfile = open(Path.joinpath(rootpath, filename), 'wb')
pickle.dump(cnn_summary_dataframe, outfile)
outfile.close()




# Testcases
print("Artikel: ", cnn_data.iloc[3,2])

print("Summary: ", cnn_data.iloc[3,3])


print("Datenstruktur: ", cnn_article_dict["Article3"])


