from pathlib import Path
import os
import pickle
import shutil
import pandas as pd
import numpy as np
import re
import nltk


# cnn_source_path = r"C:\Users\Leni\Desktop\cnn\stories_all"
# cnn_save_path = r"C:\Users\Leni\Google Drive\00_Studium\01_Master WI Goethe\01_Veranstaltungen\SS20_DS_Data Science 1\DS Project\NLP _Text Summarizer\CNN"
#
# # select 1000 files randomly
# files = os.listdir(cnn_source_path)
# random_files = np.random.choice(files, 1000)
# # save 1000 randomly selected files to shared folder (github)
# for file in random_files:
#     source_file = os.path.join(cnn_source_path, file)
#     shutil.copy(source_file, cnn_save_path)


# Relative path of dataset
os.chdir("C:/Users/Leni/Google Drive/00_Studium/01_Master WI Goethe/01_Veranstaltungen/SS20_DS_Data Science 1/DS Project/NLP _Text Summarizer/")
rootpath = Path.cwd()
cnn_path = Path.joinpath(rootpath, r"CNN")

index = np.linspace(0, 999, 1000)
cnn_data = pd.DataFrame(columns=["filename","text_raw", "text_prep", "summary"], index = index.astype(int))
cnn_data = cnn_data.fillna("nan")



count = 0
for entry in os.scandir(cnn_path):
    text_file = open(entry, "r", encoding="utf8")
    raw_text = text_file.read()
    cnn_data.iloc[count, 1] = raw_text
    cnn_data.iloc[count, 0] = str(entry)
    text_file.close()

    count += 1

# filename = "cnn_data"
# outfile = open(filename, 'wb')
# pickle.dump(cnn_data, outfile)
# outfile.close()

# filename = Path.joinpath(rootpath, r"cnn_data")
# infile = open(filename, 'rb')
# cnn_data = pickle.load(infile)
# infile.close()



not_cnn = []
empty = []

for i in range(cnn_data.shape[0]):
    cnn_data.iloc[i, 2] = re.sub("\n\n",". ",cnn_data.iloc[i,1]) ##remove new line symbol
    cnn_data.iloc[i, 2] = cnn_data.iloc[i, 2].replace("..",".")
    cnn_data.iloc[i, 2] = nltk.sent_tokenize(cnn_data.iloc[i, 2])
    if cnn_data.iloc[i, 2] != []:
        cnn_data.iloc[i, 2][0] = re.sub('^.*?-- ',"",cnn_data.iloc[i, 2][0]) #removes: (CNN) --
        cnn_data.iloc[i, 2][0] = re.sub('^.*?\(CNN\)', "", cnn_data.iloc[i, 2][0])  # removes: (CNN)
    else:
        empty.append(i)


    #print(cnn_data.iloc[i, 2])

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

    # print('summary ',cnn_data.iloc[i, 3])
    # print('article ', cnn_data.iloc[i, 2])


    if "(CNN)" not in (cnn_data.iloc[i,1]) and i not in empty:
        not_cnn.append(i)

#print(cnn_data)


##?? drop articles that are potentially from other sources // drop articles with no content
#for i in not_cnn:
    #print("not CNN?: ", cnn_data.iloc[i,2])
#for i in empty:
    #print("empty?: ", cnn_data.iloc[i,2])

cnn_data.drop(cnn_data.index[not_cnn], inplace=True)
cnn_data.drop(cnn_data.index[empty], inplace=True)
#cnn_data.reindex


## save Articles and Summaries in Dictionary, having a dataframe as item which contains one row per sentence

cnn_article_dict = {}

for i in range(cnn_data.shape[0]):
    key = "Article" + str(i)
    cnn_article_dict[key] = pd.DataFrame(cnn_data.iloc[i, 2])


cnn_summary_dict = {}

for i in range(cnn_data.shape[0]):
    key = "Summary" + str(i)
    cnn_summary_dict[key] = pd.DataFrame(cnn_data.iloc[i, 3])



# exploratory data analysis
count = 0
sentences_per_article = []
words_per_article = []
for article in cnn_article_dict.keys():
    sentences_per_article.append(cnn_article_dict[article].shape[0])

sentences_total_number = sum(sentences_per_article)



# sentences/ article on average
print("average of sentences per article: ",sentences_total_number/len(sentences_per_article))

# words/ sentence on average
print("average of words per sentence: ")

# sentences/ summary on average


# words/ summary on average


# save outputs
filename = r"cnn_articles_dict"
outfile = open(Path.joinpath(rootpath, filename), 'wb')
pickle.dump(cnn_article_dict, outfile)
outfile.close()

filename = r"cnn_summaries_dict"
outfile = open(Path.joinpath(rootpath, filename), 'wb')
pickle.dump(cnn_summary_dict, outfile)
outfile.close()

filename = r"cnn_dataframe"
outfile = open(Path.joinpath(rootpath, filename), 'wb')
pickle.dump(cnn_data, outfile)
outfile.close()






