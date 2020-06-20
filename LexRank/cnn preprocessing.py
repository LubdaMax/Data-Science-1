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
#     source_file = os.path.join(cnn_source_path,file)
#     #dest_path = os.path.join(cnn_save_path,file)
#     shutil.copy(source_file,cnn_save_path)


# Relative path of dataset
os.chdir("C:/Users/Leni/Google Drive/00_Studium/01_Master WI Goethe/01_Veranstaltungen/SS20_DS_Data Science 1/DS Project/NLP _Text Summarizer/")
rootpath = Path.cwd()
cnn_path = Path.joinpath(rootpath, r"CNN")

index = np.linspace(0, 999, 1000)
cnn_data = pd.DataFrame(columns=["filename","text_raw", "text_prep", "summary"], index = index)
cnn_data = cnn_data.fillna(0)

count = 0
for entry in os.scandir(cnn_path):
    text_file = open(entry, "r",encoding="utf8")
    raw_text = text_file.read()
    cnn_data.iloc[count, 1] = raw_text
    cnn_data.iloc[count, 0] = entry
    text_file.close()

    count += 1

for i in range(999):
    cnn_data.iloc[i,2] = re.sub("\n\n",". ",cnn_data.iloc[i,1]) ##remove new line symbol
    cnn_data.iloc[i,2] = cnn_data.iloc[i,2].replace("..",".")
    # if (cnn_data.iloc[i, 1] == ""):
    #     cnn_data = cnn_data.drop([i], axis=0)
    # cnn_data.astype(str)
    cnn_data.iloc[i, 2] = nltk.sent_tokenize(cnn_data.iloc[i, 2])
    cnn_data.iloc[i, 2][0] = re.sub('^.*?-- ',"",cnn_data.iloc[i, 2][0]) #removes: (CNN) --

    count = 0
    summary = []
    # extract summary from text
    for sentence in cnn_data.iloc[i, 2]:
        print(sentence)
        # if sentence == "@highlight.":
        #     for j in range (count,len(cnn_data.iloc[i, 2])-1):
        #         print(summary)
        #         summary.append(cnn_data.iloc[i, 2][j])
        # break

        count += 1



#print(cnn_data)
print(cnn_data.iloc[4,1])
print(cnn_data.iloc[4,2])
#print(cnn_data)
#print(cnn_data.iloc[1,1])
#print(cnn_data.iloc[1,1][0])





