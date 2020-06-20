'''
This code is used to create article and summary files from the csv file.
The output of the file will be a directory of text files representing seoarate articles and their summaries.
Each summary line starts with tag "@summary" and the article is followed by "@article".
'''
import pandas as pd
import os
import re
from pathlib import Path
import nltk
# read data from the csv file (from the location it is stored)

rootpath = Path.cwd()
filepath = Path.joinpath(rootpath, r"Wikihow\wikihowAll.csv")
Data = pd.read_csv(filepath, engine='python')
Data = Data.astype(str)
rows, columns = Data.shape

# create a file to record the file names. This can be later used to divide the dataset in train/dev/test sets
title_file = open('titles.txt', 'wb')

# The path where the articles are to be saved
savepath = Path.joinpath(rootpath, r"Wikihow\articles_including_abstract")
path = savepath
if not os.path.exists(path): os.makedirs(path)

# go over the all the articles in the data file
for row in range(rows):
    abstract = Data.loc[row,'headline']      # headline is the column representing the summary sentences
    article = Data.loc[row,'text']           # text is the column representing the article
    

    #  a threshold is used to remove short articles with long summaries as well as articles with no summary
    if len(abstract) < (0.75*len(article)):
        # remove extra commas in abstracts
        abstract = abstract.replace(".,",".")
        abstract = abstract.replace("\\n"," ")
        abstract = abstract.encode('utf-8')
        print(abstract)
        abstract_token = nltk.sent_tokenize(str(abstract))
        print(abstract_token)
        for s in abstract_token:
            
            article = re.sub(r'[.]+[\n]+[,]',"."+ s +"\n", article)
        # remove extra commas in articles
       
        article = article.encode('utf-8')
        #full = Data.loc[row, 'headline'] + Data.loc[row, 'text']
        #full = full.encode('utf-8')

        # a temporary file is created to initially write the summary, it is later used to separate the sentences of the summary
        with open('temporaryFile.txt','wb') as t:
            t.write(abstract)
        
        # file names are created using the alphanumeric charachters from the article titles.
        # they are stored in a separate text file.
        filename = Data.loc[row,'title']
        filename = "".join(x for x in filename if x.isalnum())
        filename1 = filename + '.txt'
        filename = filename.encode('utf-8')
        title_file.write(filename+b'\n')

        
        with open(str(path)+'/'+filename1,'wb') as f:
            # summary sentences will first be written into the file in separate lines
            """
            with open('temporaryFile.txt','r', encoding="utf8") as t:
                for line in t:
                    line=line.lower()
                    if line != "\n" and line != "\t" and line != " ":
                        f.write(b'@summary'+b'\n')
                        f.write(line.encode('utf-8'))
                        f.write(b'\n')
            """        
            # finally the article is written to the file
            f.write(b'@article' + b'\n')    
            f.write(article)

title_file.close()

    
